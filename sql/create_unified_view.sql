-- ============================================================
-- Create Unified Customer 360 View
-- Combines data from customers, orders, support_tickets, and analytics
-- ============================================================

CREATE OR REPLACE VIEW `conversational-360.conversational.unified_customer_view` AS

WITH customer_base AS (
    SELECT
        customer_id,
        email,
        first_name,
        last_name,
        phone,
        company,
        industry,
        acquisition_channel, -- Added Acquisition Channel
        customer_since       -- Using existing customer_since column
    FROM `conversational-360.conversational.customers`
),

customer_orders AS (
    SELECT
        c.customer_id,
        c.email,
        -- Order metrics
        COUNT(DISTINCT o.order_id) as total_orders,
        COALESCE(SUM(o.total_amount), 0) as lifetime_value,
        COALESCE(AVG(o.total_amount), 0) as avg_order_value,
        MAX(o.order_date) as last_purchase_date,
        DATE_DIFF(CURRENT_DATE(), MAX(o.order_date), DAY) as days_since_last_purchase,
        STRING_AGG(DISTINCT t.payment_method, ', ') AS preferred_payment_methods -- Added Preferred Payment
    FROM customer_base c
    LEFT JOIN `conversational-360.conversational.orders` o
        ON c.customer_id = o.customer_id
    LEFT JOIN `conversational-360.conversational.orders` t
        ON c.customer_id = t.customer_id
    GROUP BY 1,2
),

customer_support AS (
    SELECT
        customer_email as email,
        COUNT(ticket_id) as total_tickets,
        SUM(CASE WHEN status IN ('open', 'pending') THEN 1 ELSE 0 END) as open_tickets,
        
        -- Correctly calculate average resolution time in HOURS
        AVG(
            TIMESTAMP_DIFF(resolved_at, created_at, HOUR)
        ) as avg_resolution_hours,
        
        AVG(satisfaction_score) as avg_satisfaction_score,
        STRING_AGG(DISTINCT tag, ', ') AS common_ticket_tags
    FROM `conversational-360.conversational.support_tickets`
    CROSS JOIN UNNEST(SPLIT(tags, ',')) AS tag
    WHERE resolved_at IS NOT NULL AND status = 'solved' -- Only calculate for resolved tickets
    GROUP BY 1
),

customer_analytics AS (
    SELECT
        user_id as email,
        COUNT(DISTINCT session_id) as total_sessions,
        COUNT(event_id) as total_events,
        MAX(event_timestamp) as last_visit,
        ARRAY_AGG(DISTINCT traffic_source IGNORE NULLS) as acquisition_sources, -- Added Acquisition Sources
        ARRAY_AGG(DISTINCT page_url IGNORE NULLS ORDER BY event_timestamp DESC LIMIT 10) as visited_pages
    FROM `conversational-360.conversational.analytics_events`
    WHERE user_id IS NOT NULL
    GROUP BY 1
),

customer_segmentation AS (
    SELECT
        co.email,
        -- Calculate RFM-based segment based on LTV and Order Count
        CASE
            WHEN co.lifetime_value >= 10000 AND co.total_orders >= 10 THEN 'VIP'
            WHEN co.lifetime_value >= 5000 THEN 'High Value'
            WHEN co.lifetime_value >= 1000 THEN 'Medium Value'
            WHEN co.lifetime_value >= 100 THEN 'Low Value'
            ELSE 'General'
        END as customer_segment,

        -- Calculate churn risk based on recent activity and open issues
        CASE
            WHEN co.days_since_last_purchase > 180 AND COALESCE(cs.open_tickets, 0) > 1 THEN 'High Risk'
            WHEN co.days_since_last_purchase > 90 AND COALESCE(cs.open_tickets, 0) > 0 THEN 'Medium Risk'
            ELSE 'Low Risk'
        END as churn_risk
    FROM customer_orders co
    LEFT JOIN customer_support cs ON co.email = cs.email
)

-- Main unified view
SELECT
    cb.customer_id,
    cb.email,
    cb.first_name,
    cb.last_name,
    cb.phone,
    cb.company,
    cb.industry,
    cb.acquisition_channel,

    -- Order metrics
    co.total_orders,
    co.lifetime_value,
    co.avg_order_value,
    co.last_purchase_date,
    co.days_since_last_purchase,
    co.preferred_payment_methods,

    -- Support metrics
    COALESCE(cs.total_tickets, 0) as total_tickets,
    COALESCE(cs.open_tickets, 0) as open_tickets,
    cs.avg_resolution_hours,
    cs.avg_satisfaction_score,
    cs.common_ticket_tags,

    -- Analytics metrics
    COALESCE(ca.total_sessions, 0) as total_sessions,
    COALESCE(ca.total_events, 0) as total_events,
    ca.last_visit,
    ca.visited_pages,
    ca.acquisition_sources,

    -- Segmentation
    seg.customer_segment,
    seg.churn_risk,

    -- Metadata
    cb.customer_since,
    CURRENT_TIMESTAMP() as last_updated

FROM customer_base cb
LEFT JOIN customer_orders co ON cb.email = co.email
LEFT JOIN customer_support cs ON cb.email = cs.email
LEFT JOIN customer_analytics ca ON cb.email = ca.email
LEFT JOIN customer_segmentation seg ON cb.email = seg.email
WHERE cb.email IS NOT NULL;