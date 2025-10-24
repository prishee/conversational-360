#!/usr/bin/env python3
"""
Quick Fix Script - Resolves the 3 issues found in diagnostics
Run this to automatically fix your setup
"""

import os
import sys
from pathlib import Path
from google.cloud import bigquery
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
DATASET_ID = os.getenv("BQ_DATASET", "conversational")

def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def fix_unified_view():
    """Create the missing unified_customer_view"""
    print_header("FIX 1: Creating unified_customer_view")
    
    sql = f"""
CREATE OR REPLACE VIEW `{PROJECT_ID}.{DATASET_ID}.unified_customer_view` AS

WITH customer_orders AS (
  SELECT 
    c.customer_id,
    c.email,
    c.first_name,
    c.last_name,
    c.phone,
    c.created_at as customer_since,
    COUNT(DISTINCT o.order_id) as total_orders,
    COALESCE(SUM(o.total_amount), 0) as lifetime_value,
    COALESCE(AVG(o.total_amount), 0) as avg_order_value,
    MAX(o.order_date) as last_purchase_date,
    DATE_DIFF(CURRENT_DATE(), MAX(o.order_date), DAY) as days_since_last_purchase
  FROM `{PROJECT_ID}.{DATASET_ID}.customers` c
  LEFT JOIN `{PROJECT_ID}.{DATASET_ID}.orders` o 
    ON c.customer_id = o.customer_id
  GROUP BY 1,2,3,4,5,6
),

customer_support AS (
  SELECT 
    customer_email as email,
    COUNT(*) as total_tickets,
    SUM(CASE WHEN status IN ('open', 'pending') THEN 1 ELSE 0 END) as open_tickets,
    AVG(CASE 
      WHEN resolved_at IS NOT NULL 
      THEN TIMESTAMP_DIFF(resolved_at, created_at, HOUR) 
      ELSE NULL 
    END) as avg_resolution_hours,
    AVG(satisfaction_score) as avg_satisfaction_score
  FROM `{PROJECT_ID}.{DATASET_ID}.support_tickets`
  GROUP BY 1
),

customer_analytics AS (
  SELECT 
    user_id as email,
    COUNT(*) as total_sessions,
    SUM(page_views) as total_page_views,
    MAX(event_timestamp) as last_visit,
    ARRAY_AGG(DISTINCT page_url IGNORE NULLS ORDER BY event_timestamp DESC LIMIT 10) as visited_pages
  FROM `{PROJECT_ID}.{DATASET_ID}.analytics_events`
  WHERE user_id IS NOT NULL
  GROUP BY 1
),

customer_segmentation AS (
  SELECT 
    email,
    CASE 
      WHEN lifetime_value >= 10000 AND total_orders >= 10 THEN 'VIP'
      WHEN lifetime_value >= 5000 THEN 'High Value'
      WHEN lifetime_value >= 1000 THEN 'Medium Value'
      WHEN lifetime_value >= 100 THEN 'Low Value'
      ELSE 'General'
    END as customer_segment,
    CASE 
      WHEN days_since_last_purchase > 180 OR open_tickets > 2 THEN 'High Risk'
      WHEN days_since_last_purchase > 90 OR open_tickets > 0 THEN 'Medium Risk'
      ELSE 'Low Risk'
    END as churn_risk
  FROM customer_orders
)

SELECT 
  co.customer_id,
  co.email,
  co.first_name,
  co.last_name,
  co.phone,
  co.total_orders,
  co.lifetime_value,
  co.avg_order_value,
  co.last_purchase_date,
  co.days_since_last_purchase,
  COALESCE(cs.total_tickets, 0) as total_tickets,
  COALESCE(cs.open_tickets, 0) as open_tickets,
  cs.avg_resolution_hours,
  cs.avg_satisfaction_score,
  COALESCE(ca.total_sessions, 0) as total_sessions,
  COALESCE(ca.total_page_views, 0) as total_page_views,
  ca.last_visit,
  ca.visited_pages,
  seg.customer_segment,
  seg.churn_risk,
  co.customer_since,
  CURRENT_TIMESTAMP() as last_updated
FROM customer_orders co
LEFT JOIN customer_support cs ON co.email = cs.email
LEFT JOIN customer_analytics ca ON co.email = ca.email
LEFT JOIN customer_segmentation seg ON co.email = seg.email
WHERE co.email IS NOT NULL
"""
    
    try:
        client = bigquery.Client(project=PROJECT_ID)
        print(f"Creating view: {DATASET_ID}.unified_customer_view")
        
        job = client.query(sql)
        job.result()  # Wait for completion
        
        print("✅ View created successfully!")
        
        # Test it
        test_query = f"""
        SELECT COUNT(*) as customer_count
        FROM `{PROJECT_ID}.{DATASET_ID}.unified_customer_view`
        """
        result = list(client.query(test_query).result())
        count = result[0].customer_count
        print(f"✅ View is working! Found {count:,} customers")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating view: {e}")
        return False

def test_gemini_models():
    """Test different Gemini models to find one that works"""
    print_header("FIX 2: Finding working Gemini model")
    
    import vertexai
    from vertexai.preview.generative_models import GenerativeModel
    
    models = [
        "gemini-1.5-flash-002",
        "gemini-1.5-flash-001", 
        "gemini-1.5-flash",
        "gemini-pro",
        "gemini-1.5-pro-002",
        "gemini-1.5-pro-001",
        "gemini-1.5-pro"
    ]
    
    region = "us-central1"
    vertexai.init(project=PROJECT_ID, location=region)
    
    print(f"Testing models in region: {region}\n")
    
    for model_name in models:
        try:
            print(f"   Trying {model_name}...", end=" ")
            llm = GenerativeModel(model_name)
            response = llm.generate_content("Say 'test'")
            
            print("✅ WORKING!")
            print(f"\n🎉 Found working model: {model_name}")
            print(f"\nUpdate your code:")
            print(f'   self.llm_model_name = "{model_name}"')
            
            # Save to config file
            config_dir = Path("config")
            config_dir.mkdir(exist_ok=True)
            
            with open(config_dir / "model_config.txt", "w") as f:
                f.write(f"# Working Gemini Model Configuration\n")
                f.write(f"# Generated: {os.popen('date').read().strip()}\n\n")
                f.write(f"region={region}\n")
                f.write(f"model={model_name}\n")
            
            print(f"✅ Configuration saved to config/model_config.txt")
            return True, model_name
            
        except Exception as e:
            print(f"❌ Not available")
    
    print("\n❌ No working model found")
    print("\nYou may need to:")
    print("1. Enable Vertex AI API:")
    print(f"   gcloud services enable aiplatform.googleapis.com --project={PROJECT_ID}")
    print("\n2. Enable Generative AI in Console:")
    print("   https://console.cloud.google.com/vertex-ai/generative")
    
    return False, None

def update_rag_system(model_name):
    """Update the RAG system with working model"""
    print_header("FIX 3: Updating RAG system configuration")
    
    rag_file = Path("src/rag_system.py")
    
    if not rag_file.exists():
        print(f"⚠️  File not found: {rag_file}")
        return False
    
    try:
        with open(rag_file, "r") as f:
            content = f.read()
        
        # Update model name
        updated = content.replace(
            'llm_model_name: str = "gemini-1.5-pro-002"',
            f'llm_model_name: str = "{model_name}"'
        )
        
        with open(rag_file, "w") as f:
            f.write(updated)
        
        print(f"✅ Updated src/rag_system.py with model: {model_name}")
        return True
        
    except Exception as e:
        print(f"❌ Error updating file: {e}")
        return False

def verify_fixes():
    """Run a quick verification of all fixes"""
    print_header("VERIFICATION: Testing all fixes")
    
    try:
        client = bigquery.Client(project=PROJECT_ID)
        
        # Test 1: View exists and works
        print("\n1. Testing unified_customer_view...")
        query = f"SELECT COUNT(*) as cnt FROM `{PROJECT_ID}.{DATASET_ID}.unified_customer_view`"
        result = list(client.query(query).result())
        print(f"   ✅ View working ({result[0].cnt:,} customers)")
        
        # Test 2: Embeddings exist
        print("\n2. Testing embeddings...")
        query = f"""
        SELECT COUNT(*) as cnt 
        FROM `{PROJECT_ID}.{DATASET_ID}.support_tickets_embedded`
        WHERE embedding IS NOT NULL
        """
        result = list(client.query(query).result())
        print(f"   ✅ Embeddings exist ({result[0].cnt:,} documents)")
        
        # Test 3: Can query customer
        print("\n3. Testing customer query...")
        query = f"""
        SELECT email, first_name, last_name, customer_segment, churn_risk
        FROM `{PROJECT_ID}.{DATASET_ID}.unified_customer_view`
        LIMIT 1
        """
        result = list(client.query(query).result())
        if result:
            row = result[0]
            print(f"   ✅ Sample customer: {row.first_name} {row.last_name}")
            print(f"      Email: {row.email}")
            print(f"      Segment: {row.customer_segment}")
            print(f"      Churn Risk: {row.churn_risk}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Verification failed: {e}")
        return False

def main():
    print("="*70)
    print("  CONVERSATIONAL 360 - QUICK FIX SCRIPT")
    print("="*70)
    
    if not PROJECT_ID:
        print("❌ GCP_PROJECT_ID not set")
        sys.exit(1)
    
    print(f"\nProject: {PROJECT_ID}")
    print(f"Dataset: {DATASET_ID}")
    
    results = {}
    
    # Fix 1: Create unified view
    results['view'] = fix_unified_view()
    
    # Fix 2: Find working Gemini model
    results['model'], model_name = test_gemini_models()
    
    # Fix 3: Update RAG system (only if we found a model)
    if results['model']:
        results['rag'] = update_rag_system(model_name)
    else:
        results['rag'] = False
    
    # Verification
    if all([results['view'], results['model']]):
        print("\n")
        verify_fixes()
    
    # Final summary
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)
    
    print(f"\n✅ Fixed {sum(results.values())}/3 issues:")
    print(f"   {'✅' if results['view'] else '❌'} unified_customer_view created")
    print(f"   {'✅' if results['model'] else '❌'} Working Gemini model found")
    print(f"   {'✅' if results.get('rag', False) else '❌'} RAG system updated")
    
    if all(results.values()):
        print("\n🎉 ALL FIXES APPLIED! You can now run:")
        print("   streamlit run app.py")
    else:
        print("\n⚠️  Some fixes failed. Review the output above for details.")
    
    print("="*70)

if __name__ == "__main__":
    main()