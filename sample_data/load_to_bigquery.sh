#!/bin/bash
# Load sample data into BigQuery

# 1. Update with your actual PROJECT_ID and DATASET
PROJECT_ID="conversational-360"
DATASET="conversational"

echo "Loading data into BigQuery..."

# 2. Update table names to match Fivetran's conventions (e.g., 'customers' instead of 'salesforce_Contact')

# Load customers
# Mapped to 'conversational.customers'
bq load --source_format=CSV --skip_leading_rows=1 \
  $PROJECT_ID:$DATASET.customers \
  sample_data/customers.csv \
  customer_id:STRING,email:STRING,first_name:STRING,last_name:STRING,phone:STRING,company:STRING,industry:STRING,customer_since:TIMESTAMP,acquisition_channel:STRING,account_status:STRING

# Load products
# Mapped to 'conversational.products'
bq load --source_format=CSV --skip_leading_rows=1 \
  $PROJECT_ID:$DATASET.products \
  sample_data/products.csv \
  product_id:STRING,product_name:STRING,description:STRING,category:STRING,price:FLOAT,stock_quantity:INTEGER,rating:FLOAT,created_at:TIMESTAMP

# Load orders
# Mapped to 'conversational.orders'
bq load --source_format=CSV --skip_leading_rows=1 \
  $PROJECT_ID:$DATASET.orders \
  sample_data/orders.csv \
  order_id:STRING,customer_id:STRING,customer_email:STRING,order_date:TIMESTAMP,status:STRING,total_amount:FLOAT,num_items:INTEGER,shipping_address:STRING,payment_method:STRING

# Load support tickets
# Mapped to 'conversational.support_tickets'
bq load --source_format=CSV --skip_leading_rows=1 \
  $PROJECT_ID:$DATASET.support_tickets \
  sample_data/support_tickets.csv \
  ticket_id:STRING,customer_id:STRING,customer_email:STRING,subject:STRING,description:STRING,status:STRING,priority:STRING,created_at:TIMESTAMP,updated_at:TIMESTAMP,resolution_time_hours:INTEGER,satisfaction_rating:INTEGER,tags:STRING

# Load analytics events
# Mapped to 'conversational.analytics_events'
bq load --source_format=CSV --skip_leading_rows=1 \
  $PROJECT_ID:$DATASET.analytics_events \
  sample_data/analytics_events.csv \
  event_id:STRING,user_id:STRING,session_id:STRING,event_type:STRING,page_path:STRING,event_timestamp:TIMESTAMP,device_category:STRING,traffic_source:STRING

echo "Data loaded successfully!"