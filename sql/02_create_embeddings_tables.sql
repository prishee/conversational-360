-- ============================================================
-- EMBEDDING TABLES - FINAL FIXED VERSION
-- ============================================================

-- Support tickets embedding table
CREATE TABLE IF NOT EXISTS `conversational.support_tickets_embedded` (
  ticket_id STRING,
  customer_email STRING,
  subject STRING,
  description STRING,
  status STRING,
  priority STRING,
  created_at TIMESTAMP,
  updated_at TIMESTAMP,
  satisfaction_score FLOAT64,
  full_text STRING,
  embedding ARRAY<FLOAT64>,
  embedding_model STRING,
  embedded_at TIMESTAMP
)
PARTITION BY DATE(created_at)
CLUSTER BY customer_email, status;

-- Product catalog embedding table
CREATE TABLE IF NOT EXISTS `conversational.product_catalog_embedded` (
  product_id STRING,
  product_name STRING,
  description STRING,
  category STRING,
  price FLOAT64,
  full_text STRING,
  embedding ARRAY<FLOAT64>,
  embedding_model STRING,
  embedded_at TIMESTAMP
);

-- Statement 3: Populate support tickets (FIXED with explicit CAST)
MERGE INTO `conversational.support_tickets_embedded` T
USING (
  SELECT 
    ticket_id,
    customer_email,
    subject,
    description,
    status,
    priority,
    -- FIX: Explicitly CAST DATETIME to TIMESTAMP
    CAST(created_at AS TIMESTAMP) AS created_at, 
    CAST(updated_at AS TIMESTAMP) AS updated_at,
    satisfaction_rating AS satisfaction_score,
    CONCAT(
      'Subject: ', COALESCE(subject, ''), '\n',
      'Description: ', COALESCE(description, ''), '\n',
      'Status: ', COALESCE(status, ''), '\n',
      'Priority: ', COALESCE(priority, '')
    ) AS full_text
  FROM `conversational.support_tickets`
) S
ON T.ticket_id = S.ticket_id
WHEN NOT MATCHED THEN
  INSERT (ticket_id, customer_email, subject, description, status, priority, 
          created_at, updated_at, satisfaction_score, full_text)
  VALUES (S.ticket_id, S.customer_email, S.subject, S.description, S.status, 
          S.priority, S.created_at, S.updated_at, S.satisfaction_score, S.full_text);

-- Statement 4: Populate products
MERGE INTO `conversational.product_catalog_embedded` T
USING (
  SELECT 
    product_id,
    product_name,
    description,
    category,
    price,
    CONCAT(
      'Product: ', product_name, '\n',
      'Category: ', COALESCE(category, 'General'), '\n',
      'Description: ', COALESCE(description, ''), '\n',
      'Price: $', CAST(price AS STRING)
    ) AS full_text
  FROM `conversational.products`
) S
ON T.product_id = S.product_id
WHEN NOT MATCHED THEN
  INSERT (product_id, product_name, description, category, price, full_text)
  VALUES (S.product_id, S.product_name, S.description, S.category, S.price, S.full_text);