import pandas as pd
import numpy as np
from faker import Faker
import random
import os
from datetime import timedelta, datetime
import json

# Initialize the Faker library
fake = Faker()

# Define categories of alcohol and sample product names
categories = {
    'Beer': ['Lager', 'IPA', 'Stout', 'Pilsner', 'Ale', 'Porter', 'Wheat Beer', 'Sour Beer', 'Pale Ale', 'Brown Ale'],
    'Wine': ['Merlot', 'Chardonnay', 'Pinot Noir', 'Sauvignon Blanc', 'Cabernet Sauvignon', 'Zinfandel', 'Rose', 'Sparkling', 'Syrah', 'Riesling'],
    'Whiskey': ['Bourbon', 'Scotch', 'Irish Whiskey', 'Rye Whiskey', 'Tennessee Whiskey', 'Japanese Whiskey', 'Single Malt', 'Blended Whiskey', 'Canadian Whiskey', 'Corn Whiskey'],
    'Vodka': ['Classic Vodka', 'Flavored Vodka', 'Premium Vodka', 'Grain Vodka', 'Potato Vodka', 'Corn Vodka', 'Organic Vodka', 'Gluten-Free Vodka', 'Citrus Vodka', 'Herb Vodka'],
    'Rum': ['White Rum', 'Dark Rum', 'Spiced Rum', 'Gold Rum', 'Aged Rum', 'Overproof Rum', 'Rhum Agricole', 'Flavored Rum', 'Premium Rum', 'Light Rum'],
    'Tequila': ['Blanco', 'Reposado', 'Anejo', 'Extra Anejo', 'Cristalino', 'Joven', 'Gold Tequila', 'Silver Tequila', '100% Agave', 'Mixto'],
    'Gin': ['London Dry Gin', 'Plymouth Gin', 'Old Tom Gin', 'Genever', 'New American Gin', 'Sloe Gin', 'Flavored Gin', 'Navy Strength Gin', 'Botanical Gin', 'Craft Gin']
}

# List of supplier companies
companies = ['Global Spirits', 'Fine Wine Suppliers', 'Beer Distributors Inc.', 'Whiskey Wholesalers', 'Premium Liquor Co.']

# Generate initial stock data
stock_data = []

for category, products in categories.items():
    for product in products:
        base_price = round(random.uniform(10, 200), 2)  # Random base price between $10 and $200
        for _ in range(10):
            name = f"{product} {category}"
            # Ensure price difference does not exceed 5%
            price_variation = base_price * 0.05
            price = round(random.uniform(base_price - price_variation, base_price + price_variation), 2)
            order_date = fake.date_this_year()
            delivery_days = random.randint(1, 14)  # Delivery date varies from 1 day to 2 weeks
            delivered_date = order_date + timedelta(days=delivery_days)
            company = random.choice(companies)
            stock = random.randint(3, 20)  # Stock quantity between 3 and 20 bottles
            stock_data.append([name, price, order_date.strftime("%Y-%m-%d"), delivered_date.strftime("%Y-%m-%d"), company, stock])

# Create stock DataFrame
stock_df = pd.DataFrame(stock_data, columns=['Name', 'Price', 'Order Date', 'Delivered Date', 'Company', 'Stock'])

# Simulate daily consumption and reorder process
consumption_days = 30
consumption_data = []
order_data = []
price_dict = {}  # Dictionary to hold the prices for products from each supplier
order_counter = 1

for i, row in stock_df.iterrows():
    current_stock = row['Stock']
    product = row['Name']
    initial_price = row['Price']
    supplier = row['Company']
    price_dict[(product, supplier)] = initial_price  # Set initial price for each product-supplier pair
    for day in range(consumption_days):
        date = datetime.now() - timedelta(days=(consumption_days - day))
        if current_stock > 0:
            consumption = random.randint(0, min(2, current_stock))  # Random consumption up to 2 bottles, but not more than available stock
            current_stock -= consumption
        else:
            consumption = 0

        consumption_data.append([product, date.strftime("%Y-%m-%d"), consumption, current_stock])

        # Simulate reorder when stock is low or depleted
        if current_stock == 0 or random.random() < 0.1:
            order_date = date
            delivery_days = random.randint(1, 14)
            delivered_date = order_date + timedelta(days=delivery_days)
            stock_replenishment = random.randint(3, 20)
            current_stock += stock_replenishment
            company = supplier  # Reorder from the same supplier
            price = price_dict[(product, supplier)]  # Use the same price for the same product from the same supplier
            order_data.append([order_counter, product, price, order_date.strftime("%Y-%m-%d"), delivered_date.strftime("%Y-%m-%d"), company, stock_replenishment])
            order_counter += 1

# Create consumption DataFrame
consumption_df = pd.DataFrame(consumption_data, columns=['Name', 'Date', 'Consumption', 'Remaining Stock'])

# Create order DataFrame
order_df = pd.DataFrame(order_data, columns=['Order Number', 'Name', 'Price', 'Order Date', 'Delivered Date', 'Company', 'Stock'])

# Summarize stock per category
summarized_stock = stock_df.groupby('Name').agg({
    'Price': 'mean',
    'Order Date': 'max',
    'Delivered Date': 'max',
    'Stock': 'sum'
}).reset_index()

# Save each DataFrame to a separate JSON file
summarized_stock_file = os.path.join(os.getcwd(), "summarized_stock_data.json")
consumption_file = os.path.join(os.getcwd(), "consumption_data.json")
orders_file = os.path.join(os.getcwd(), "orders_data.json")

summarized_stock.to_json(summarized_stock_file, orient='records', indent=4)
consumption_df.to_json(consumption_file, orient='records', indent=4)
order_df.to_json(orders_file, orient='records', indent=4)

# Display the paths to the saved files
print(f"Summarized stock data saved to {summarized_stock_file}")
print(f"Consumption data saved to {consumption_file}")
print(f"Orders data saved to {orders_file}")
