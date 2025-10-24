-- ============================================================
-- FINAL UNIFIED CUSTOMER 360 VIEW - FIXED COLUMN NAMES
-- ============================================================

CREATE OR REPLACE VIEW `conversational.unified_customer_view` AS
WITH customer_base AS (
  SELECT 
    customer_id,
    email,
    first_name,
    last_name,
    phone,
    company,
    industry,
    customer_since,
    acquisition_channel,
    account_status
  FROM `conversational.customers`
  WHERE email IS NOT NULL
),

purchase_metrics AS (
  SELECT 
    customer_email AS email,
    COUNT(DISTINCT order_id) AS total_orders,
    SUM(total_amount) AS lifetime_value,
    MAX(order_date) AS last_purchase_date,
    AVG(total_amount) AS avg_order_value
  FROM `conversational.orders`
  WHERE customer_email IS NOT NULL
  GROUP BY customer_email
),

support_metrics AS (
  SELECT 
    customer_email AS email,
    COUNT(DISTINCT ticket_id) AS total_tickets,
    COUNT(DISTINCT CASE WHEN status IN ('open', 'pending') THEN ticket_id END) AS open_tickets,
    AVG(resolution_time_hours) AS avg_resolution_hours,
    MAX(created_at) AS last_ticket_date,
    AVG(satisfaction_rating) AS avg_satisfaction_score
  FROM `conversational.support_tickets`
  WHERE customer_email IS NOT NULL
  GROUP BY customer_email
),

analytics_metrics AS (
  SELECT
    user_id AS customer_id,
    COUNT(DISTINCT session_id) AS total_sessions,
    COUNT(DISTINCT event_id) AS total_page_views,
    MAX(event_timestamp) AS last_visit,
    ARRAY_AGG(DISTINCT page_path IGNORE NULLS LIMIT 10) AS visited_pages
  FROM `conversational.analytics_events`
  WHERE user_id IS NOT NULL
  GROUP BY user_id
)

SELECT 
  cb.customer_id,
  cb.email,
  cb.first_name,
  cb.last_name,
  cb.phone,
  cb.company AS account_name,
  cb.industry,
  cb.customer_since,
  cb.acquisition_channel,
  
  -- Purchase metrics
  COALESCE(pm.total_orders, 0) AS total_orders,
  COALESCE(pm.lifetime_value, 0) AS lifetime_value,
  pm.last_purchase_date,
  COALESCE(pm.avg_order_value, 0) AS avg_order_value,
  ARRAY<STRING>[] AS purchased_product_ids,
  
  -- Support metrics
  COALESCE(sm.total_tickets, 0) AS total_tickets,
  COALESCE(sm.open_tickets, 0) AS open_tickets,
  sm.avg_resolution_hours,
  sm.last_ticket_date,
  sm.avg_satisfaction_score,
  
  -- Analytics metrics
  COALESCE(am.total_sessions, 0) AS total_sessions,
  COALESCE(am.total_page_views, 0) AS total_page_views,
  am.last_visit,
  COALESCE(am.visited_pages, ARRAY<STRING>[]) AS visited_pages,
  
  -- Calculated fields
  DATE_DIFF(CURRENT_DATE(), DATE(pm.last_purchase_date), DAY) AS days_since_last_purchase,
  
  -- Customer segment
  CASE 
    WHEN pm.lifetime_value > 10000 THEN 'VIP'
    WHEN pm.lifetime_value > 5000 THEN 'High Value'
    WHEN pm.lifetime_value > 1000 THEN 'Medium Value'
    ELSE 'Low Value'
  END AS customer_segment,
  
  -- Churn risk
  CASE 
    WHEN DATE_DIFF(CURRENT_DATE(), DATE(pm.last_purchase_date), DAY) > 180 
      AND sm.open_tickets > 0 THEN 'High Risk'
    WHEN DATE_DIFF(CURRENT_DATE(), DATE(pm.last_purchase_date), DAY) > 90 THEN 'Medium Risk'
    ELSE 'Low Risk'
  END AS churn_risk

FROM customer_base cb
LEFT JOIN purchase_metrics pm ON cb.email = pm.email
LEFT JOIN support_metrics sm ON cb.email = sm.email
LEFT JOIN analytics_metrics am ON cb.customer_id = am.customer_id;