import pandas as pd
import json
import os
import logging
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryResult
from llama_index.core.schema import NodeWithScore
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from llamaapi import LlamaAPI
import asyncio
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the SDK
llama = LlamaAPI("LL-0HPBPQGQjMaZRmsKpoJtTySoccREMLIHWsYTW5gVYU2y3fMl36WauGCYuJt5MAAn")

# Load JSON Data
try:
    summarized_stock_df = pd.read_json("summarized_stock_data.json")
    consumption_df = pd.read_json("consumption_data.json")
    order_df = pd.read_json("orders_data.json")
except FileNotFoundError as e:
    logging.error(f"Error loading data: {e}")
    raise

# Initialize Sentence-Transformers Model
model = SentenceTransformer('all-MiniLM-L6-v2')


def generate_embeddings(dataframe, column):
    return np.array([model.encode(str(text)) for text in dataframe[column]])


def save_embeddings(file_path, embeddings):
    np.save(file_path, embeddings)


def load_embeddings(file_path):
    return np.load(file_path)


def create_and_save_faiss_index(embeddings, file_path):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, file_path)
    return index


def load_faiss_index(file_path):
    return faiss.read_index(file_path)


# Paths for cached embeddings and indices
summarized_stock_embeddings_path = "summarized_stock_embeddings.npy"
consumption_embeddings_path = "consumption_embeddings.npy"
order_embeddings_path = "order_embeddings.npy"
stock_index_path = "stock_index.faiss"
consumption_index_path = "consumption_index.faiss"
order_index_path = "order_index.faiss"

# Generate or load embeddings and create/load FAISS indices
try:
    if not os.path.exists(stock_index_path):
        logging.info("Creating and saving stock_index")
        summarized_stock_embeddings = generate_embeddings(summarized_stock_df, 'Name')
        stock_index = create_and_save_faiss_index(summarized_stock_embeddings, stock_index_path)
    else:
        logging.info("Loading cached stock_index")
        stock_index = load_faiss_index(stock_index_path)

    if not os.path.exists(consumption_index_path):
        logging.info("Creating and saving consumption_index")
        consumption_embeddings = generate_embeddings(consumption_df, 'Name')
        consumption_index = create_and_save_faiss_index(consumption_embeddings, consumption_index_path)
    else:
        logging.info("Loading cached consumption_index")
        consumption_index = load_faiss_index(consumption_index_path)

    if not os.path.exists(order_index_path):
        logging.info("Creating and saving order_index")
        order_embeddings = generate_embeddings(order_df, 'Name')
        order_index = create_and_save_faiss_index(order_embeddings, order_index_path)
    else:
        logging.info("Loading cached order_index")
        order_index = load_faiss_index(order_index_path)
except Exception as e:
    logging.error(f"Error in index creation or loading: {e}")
    raise

# Initialize Vector Stores
stock_vector_store = FaissVectorStore(faiss_index=stock_index)
consumption_vector_store = FaissVectorStore(faiss_index=consumption_index)
order_vector_store = FaissVectorStore(faiss_index=order_index)

# Updated prompt for query refinement
refine_template = """Refine the following query to make it specific and clear for a vector-based search. Focus on key terms and concepts that would be relevant for semantic similarity.

Original Query: {query}

Refined Query for Vector Search:
"""

# Updated prompt for interpreting vector search results
interpret_template = """Using the provided vector search results, generate a clear and concise answer to the original query. The answer should be based on the semantic similarity of the results to the query.

Original Query: {query}

Vector Search Results: {search_results}

Answer:
"""


def parse_llama_response(response):
    try:
        return response.json()
    except json.JSONDecodeError:
        logging.error("Failed to parse response content as JSON")
        return None





def vector_search(vector_store, query: str, df: pd.DataFrame, top_k: int = 10) -> List[Dict[str, Any]]:
    try:
        query_embedding = model.encode(query)

        # Convert the embedding to a list of floats
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
        elif hasattr(query_embedding, 'cpu'):  # Check if it's a torch.Tensor
            query_embedding = query_embedding.cpu().numpy().tolist()
        else:
            query_embedding = [float(x) for x in query_embedding]

        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=top_k,
            query_str=query
        )

        logging.info(f"Querying vector store with query: {query}")
        results: VectorStoreQueryResult = vector_store.query(vector_store_query)
        logging.info(f"Received results from vector store: {results}")

        if results is None or (results.nodes is None and results.ids is None):
            logging.warning("Vector store returned no usable results")
            return []

        processed_results = []
        if results.nodes is not None:
            for node, score in zip(results.nodes, results.similarities or []):
                if node is None:
                    logging.warning("Encountered None node in results")
                    continue
                try:
                    processed_results.append({
                        "content": node.get_content() if hasattr(node, 'get_content') else str(node),
                        "metadata": node.metadata if hasattr(node, 'metadata') else {},
                        "score": float(score) if score is not None else None
                    })
                except Exception as e:
                    logging.error(f"Error processing node: {e}")
        elif results.ids is not None:
            for id, score in zip(results.ids, results.similarities or []):
                try:
                    node_data = df.loc[df.index == int(id)].iloc[0].to_dict() if int(id) in df.index else None
                    if node_data:
                        processed_results.append({
                            "content": str(node_data),
                            "metadata": node_data,
                            "score": float(score) if score is not None else None
                        })
                except Exception as e:
                    logging.error(f"Error processing ID {id}: {e}")

        return processed_results

    except Exception as e:
        logging.error(f"Error in vector_search: {e}")
        return []


async def handle_query(query: str):
    try:
        # Step 1: Refine the query
        refine_request_json = {
            "model": "llama3.1-405b",
            "messages": [
                {
                    "role": "user",
                    "content": refine_template.format(query=query)
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.7
        }

        logging.info(f"Sending request to LlamaAPI for query refinement: {query}")
        refine_response = llama.run_sync(refine_request_json)
        logging.info(f"Received refined query from LlamaAPI: {refine_response}")

        parsed_refine_response = parse_llama_response(refine_response)
        if parsed_refine_response is None or 'choices' not in parsed_refine_response:
            logging.warning("Failed to parse refine response, using original query")
            refined_query = query
        else:
            refined_query = parsed_refine_response['choices'][0]['message']['content']

        # Step 2: Perform vector search
        logging.info(f"Performing vector search with refined query: {refined_query}")
        stock_results = vector_search(stock_vector_store, refined_query, summarized_stock_df)
        consumption_results = vector_search(consumption_vector_store, refined_query, consumption_df)
        order_results = vector_search(order_vector_store, refined_query, order_df)

        all_results = stock_results + consumption_results + order_results
        all_results.sort(key=lambda x: x['score'] if x['score'] is not None else float('-inf'), reverse=True)
        top_results = all_results[:10]  # Get top 10 results across all stores

        logging.info(f"Top results: {top_results}")

        # Step 3: Interpret the search results
        interpret_request_json = {
            "model": "llama3.1-405b",
            "messages": [
                {
                    "role": "user",
                    "content": interpret_template.format(
                        query=query,
                        search_results=json.dumps(top_results, indent=2)
                    )
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.7
        }

        logging.info(f"Sending request to LlamaAPI for result interpretation: {query}")
        interpret_response = llama.run_sync(interpret_request_json)
        logging.info(f"Received interpretation from LlamaAPI: {interpret_response}")

        parsed_interpret_response = parse_llama_response(interpret_response)
        if parsed_interpret_response is None or 'choices' not in parsed_interpret_response:
            logging.warning("Failed to interpret results, returning raw search results")
            return top_results
        else:
            interpretation = parsed_interpret_response['choices'][0]['message']['content']
            return interpretation

    except Exception as e:
        logging.error(f"Error handling query: {e}")
        return f"An error occurred while processing your query: {str(e)}"


async def chat():
    print("Welcome to the Stock Inventory Chat!")
    print("You can ask questions about stock levels, consumption, and orders.")
    print("Type 'exit' to end the chat.")

    while True:
        user_query = input("\nYou: ")
        if user_query.lower() == 'exit':
            print("Thank you for using the Stock Inventory Chat. Goodbye!")
            break

        print("Processing your query...")
        result = await handle_query(user_query)
        print(f"\nAssistant: {result}")


if __name__ == "__main__":
    asyncio.run(chat())
