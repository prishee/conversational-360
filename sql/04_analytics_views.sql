-- ============================================================
-- ANALYTICS VIEWS
-- ============================================================

CREATE OR REPLACE VIEW `conversational.churn_risk_summary` AS
SELECT 
  churn_risk,
  COUNT(*) AS customer_count,
  AVG(lifetime_value) AS avg_ltv,
  AVG(days_since_last_purchase) AS avg_days_inactive
FROM `conversational.unified_customer_view`
GROUP BY churn_risk;

CREATE OR REPLACE VIEW `conversational.customer_segments` AS
SELECT 
  customer_segment,
  COUNT(*) AS customer_count,
  AVG(lifetime_value) AS avg_ltv,
  AVG(total_orders) AS avg_orders
FROM `conversational.unified_customer_view`
GROUP BY customer_segment;