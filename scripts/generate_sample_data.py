#!/usr/bin/env python3
"""
Generate realistic sample data for Conversational 360 demo
Creates CSV files that can be loaded into BigQuery or connected via Fivetran
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from faker import Faker
import random

fake = Faker()
Faker.seed(42)
np.random.seed(42)

# Configuration
NUM_CUSTOMERS = 500
NUM_ORDERS = 2000
NUM_TICKETS = 1500
NUM_PRODUCTS = 50

print(" Generating realistic sample data...\n")

# ============================================================
# 1. GENERATE CUSTOMERS (Salesforce-like CRM data)
# ============================================================

print(" Generating customers...")

customers = []
for i in range(NUM_CUSTOMERS):
    created_date = fake.date_time_between(start_date='-3y', end_date='-6m')
    
    customer = {
        'customer_id': f'CUST-{i+1000:05d}',
        'email': fake.email(),
        'first_name': fake.first_name(),
        'last_name': fake.last_name(),
        'phone': fake.phone_number(),
        'company': fake.company() if random.random() > 0.3 else None,
        'industry': random.choice(['Technology', 'Retail', 'Healthcare', 'Finance', 'Manufacturing']),
        'customer_since': created_date.isoformat(),
        'acquisition_channel': random.choice(['Organic Search', 'Paid Ads', 'Referral', 'Direct', 'Social Media']),
        'account_status': random.choice(['Active', 'Active', 'Active', 'Inactive']),
    }
    customers.append(customer)

customers_df = pd.DataFrame(customers)
print(f"    Created {len(customers_df)} customers")

# ============================================================
# 2. GENERATE PRODUCTS (Shopify-like catalog)
# ============================================================

print(" Generating products...")

product_categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books']
products = []

for i in range(NUM_PRODUCTS):
    category = random.choice(product_categories)
    products.append({
        'product_id': f'PROD-{i+100:04d}',
        'product_name': fake.catch_phrase(),
        'description': fake.text(200),
        'category': category,
        'price': round(random.uniform(9.99, 499.99), 2),
        'stock_quantity': random.randint(0, 500),
        'rating': round(random.uniform(3.0, 5.0), 1),
        'created_at': fake.date_time_between(start_date='-2y', end_date='-1y').isoformat()
    })

products_df = pd.DataFrame(products)
print(f"    Created {len(products_df)} products")

# ============================================================
# 3. GENERATE ORDERS (Shopify-like transactions)
# ============================================================

print("  Generating orders...")

orders = []
order_statuses = ['completed', 'completed', 'completed', 'completed', 'pending', 'cancelled']

for i in range(NUM_ORDERS):
    customer = customers_df.sample(1).iloc[0]
    customer_created = pd.to_datetime(customer['customer_since'])
    
    # Orders happen after customer creation
    order_date = fake.date_time_between(
        start_date=customer_created,
        end_date='now'
    )
    
    num_items = random.randint(1, 5)
    order_products = products_df.sample(num_items)
    total = sum(order_products['price'] * np.random.randint(1, 4, num_items))
    
    orders.append({
        'order_id': f'ORD-{i+10000:06d}',
        'customer_id': customer['customer_id'],
        'customer_email': customer['email'],
        'order_date': order_date.isoformat(),
        'status': random.choice(order_statuses),
        'total_amount': round(total, 2),
        'num_items': num_items,
        'shipping_address': fake.address(),
        'payment_method': random.choice(['Credit Card', 'PayPal', 'Debit Card'])
    })

orders_df = pd.DataFrame(orders)
print(f"    Created {len(orders_df)} orders")

# ============================================================
# 4. GENERATE SUPPORT TICKETS (Zendesk-like)
# ============================================================

print(" Generating support tickets...")

ticket_subjects = [
    "Product not working as expected",
    "Billing question about recent charge",
    "Shipping delay inquiry",
    "Request for refund",
    "Technical issue with product",
    "Missing items from order",
    "Account access problem",
    "Product recommendation request",
    "Complaint about quality",
    "Question about warranty"
]

ticket_descriptions = {
    "Product not working": "I received the product but it's not functioning properly. I've tried troubleshooting but the issue persists.",
    "Billing question": "I noticed a charge on my account that I don't recognize. Can you please explain what this is for?",
    "Shipping delay": "My order was supposed to arrive 3 days ago but I haven't received it yet. Can you check the status?",
    "Request for refund": "I'd like to return this product and get a full refund. It doesn't meet my expectations.",
    "Technical issue": "I'm experiencing technical difficulties with the product. Error code 404 keeps appearing.",
    "Missing items": "I received my order but some items are missing from the package. Order #{order_id}",
    "Account access": "I can't log into my account anymore. Password reset isn't working.",
    "Product recommendation": "I'm looking for a product similar to what I purchased before. Can you recommend something?",
    "Quality complaint": "The product quality is very poor. It broke after just 2 days of use.",
    "Warranty question": "What's the warranty policy on this product? I might need to make a claim."
}

tickets = []
ticket_statuses = ['open', 'open', 'pending', 'solved', 'solved', 'solved', 'solved']
priorities = ['low', 'low', 'medium', 'medium', 'high']

for i in range(NUM_TICKETS):
    customer = customers_df.sample(1).iloc[0]
    customer_created = pd.to_datetime(customer['customer_since'])
    
    created_date = fake.date_time_between(
        start_date=customer_created,
        end_date='now'
    )
    
    status = random.choice(ticket_statuses)
    subject = random.choice(ticket_subjects)
    
    # If solved, add resolution time
    resolution_time = None
    satisfaction = None
    if status == 'solved':
        resolution_time = random.randint(1, 72)  # hours
        satisfaction = random.choice([3, 4, 4, 5, 5, 5])  # Skewed positive
    
    tickets.append({
        'ticket_id': f'TKT-{i+5000:05d}',
        'customer_id': customer['customer_id'],
        'customer_email': customer['email'],
        'subject': subject,
        'description': ticket_descriptions.get(subject.split()[0], "Customer inquiry requiring attention."),
        'status': status,
        'priority': random.choice(priorities),
        'created_at': created_date.isoformat(),
        'updated_at': (created_date + timedelta(hours=random.randint(1, 48))).isoformat(),
        'resolution_time_hours': resolution_time,
        'satisfaction_rating': satisfaction,
        'tags': '|'.join(random.sample(['billing', 'shipping', 'product', 'technical', 'refund'], k=random.randint(1, 3)))
    })

tickets_df = pd.DataFrame(tickets)
print(f"    Created {len(tickets_df)} support tickets")

# ============================================================
# 5. GENERATE ANALYTICS EVENTS (Google Analytics-like)
# ============================================================

print(" Generating analytics events...")

events = []
event_types = ['page_view', 'page_view', 'page_view', 'add_to_cart', 'purchase', 'search']
pages = ['/home', '/products', '/product/{id}', '/cart', '/checkout', '/account', '/support']

for i in range(NUM_CUSTOMERS * 50):  # ~50 events per customer
    customer = customers_df.sample(1).iloc[0]
    customer_created = pd.to_datetime(customer['customer_since'])
    
    event_time = fake.date_time_between(
        start_date=customer_created,
        end_date='now'
    )
    
    events.append({
        'event_id': f'EVT-{i:08d}',
        'user_id': customer['customer_id'],
        'session_id': f'SES-{random.randint(1000, 9999):04d}',
        'event_type': random.choice(event_types),
        'page_path': random.choice(pages),
        'event_timestamp': event_time.isoformat(),
        'device_category': random.choice(['desktop', 'mobile', 'tablet']),
        'traffic_source': customer['acquisition_channel']
    })

events_df = pd.DataFrame(events)
print(f"    Created {len(events_df)} analytics events")

# ============================================================
# 6. SAVE TO CSV FILES
# ============================================================

print("\n Saving data to CSV files...")

output_dir = 'sample_data'
import os
os.makedirs(output_dir, exist_ok=True)

customers_df.to_csv(f'{output_dir}/customers.csv', index=False)
products_df.to_csv(f'{output_dir}/products.csv', index=False)
orders_df.to_csv(f'{output_dir}/orders.csv', index=False)
tickets_df.to_csv(f'{output_dir}/support_tickets.csv', index=False)
events_df.to_csv(f'{output_dir}/analytics_events.csv', index=False)

print(f"    Files saved to {output_dir}/")

# ============================================================
# 7. GENERATE LOAD SCRIPT FOR BIGQUERY
# ============================================================

print("\n Generating BigQuery load script...")

load_script = f"""#!/bin/bash
# Load sample data into BigQuery

PROJECT_ID="{os.getenv('GCP_PROJECT_ID', 'your-project-id')}"
DATASET="customer_360"

echo "Loading data into BigQuery..."

# Load customers
bq load --source_format=CSV --skip_leading_rows=1 \\
  $PROJECT_ID:$DATASET.salesforce_Contact \\
  sample_data/customers.csv \\
  customer_id:STRING,email:STRING,first_name:STRING,last_name:STRING,phone:STRING,company:STRING,industry:STRING,customer_since:TIMESTAMP,acquisition_channel:STRING,account_status:STRING

# Load products
bq load --source_format=CSV --skip_leading_rows=1 \\
  $PROJECT_ID:$DATASET.shopify_product \\
  sample_data/products.csv \\
  product_id:STRING,product_name:STRING,description:STRING,category:STRING,price:FLOAT,stock_quantity:INTEGER,rating:FLOAT,created_at:TIMESTAMP

# Load orders
bq load --source_format=CSV --skip_leading_rows=1 \\
  $PROJECT_ID:$DATASET.shopify_order \\
  sample_data/orders.csv \\
  order_id:STRING,customer_id:STRING,customer_email:STRING,order_date:TIMESTAMP,status:STRING,total_amount:FLOAT,num_items:INTEGER,shipping_address:STRING,payment_method:STRING

# Load support tickets
bq load --source_format=CSV --skip_leading_rows=1 \\
  $PROJECT_ID:$DATASET.zendesk_tickets \\
  sample_data/support_tickets.csv \\
  ticket_id:STRING,customer_id:STRING,customer_email:STRING,subject:STRING,description:STRING,status:STRING,priority:STRING,created_at:TIMESTAMP,updated_at:TIMESTAMP,resolution_time_hours:INTEGER,satisfaction_rating:INTEGER,tags:STRING

# Load analytics events
bq load --source_format=CSV --skip_leading_rows=1 \\
  $PROJECT_ID:$DATASET.google_analytics_events \\
  sample_data/analytics_events.csv \\
  event_id:STRING,user_id:STRING,session_id:STRING,event_type:STRING,page_path:STRING,event_timestamp:TIMESTAMP,device_category:STRING,traffic_source:STRING

echo " Data loaded successfully!"
"""

with open(f'{output_dir}/load_to_bigquery.sh', 'w') as f:
    f.write(load_script)

os.chmod(f'{output_dir}/load_to_bigquery.sh', 0o755)

print(f"    BigQuery load script created")

# ============================================================
# 8. SUMMARY STATISTICS
# ============================================================

print("\n" + "="*60)
print(" SUMMARY STATISTICS")
print("="*60)
print(f"Total Customers: {len(customers_df):,}")
print(f"Total Products: {len(products_df):,}")
print(f"Total Orders: {len(orders_df):,}")
print(f"Total Support Tickets: {len(tickets_df):,}")
print(f"Total Analytics Events: {len(events_df):,}")
print(f"\nRevenue (Total Orders): ${orders_df['total_amount'].sum():,.2f}")
print(f"Average Order Value: ${orders_df['total_amount'].mean():.2f}")
print(f"Open Tickets: {len(tickets_df[tickets_df['status'] == 'open']):,}")
print(f"Customer Satisfaction (Avg): {tickets_df['satisfaction_rating'].mean():.2f}/5.0")
print("="*60)

print("\n Sample data generation complete!")
print(f"\n Files created in '{output_dir}/' directory")
print(f"\n Next steps:")
print(f"   1. Run: bash {output_dir}/load_to_bigquery.sh")
print(f"   2. Or upload CSVs to Google Sheets and connect via Fivetran")
print(f"   3. Or use CSV file connector in Fivetran")