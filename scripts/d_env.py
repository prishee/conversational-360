#!/usr/bin/env python3
"""
Diagnostic script to verify BigQuery and Vertex AI setup
Run this to identify configuration issues
"""

import os
import sys
from google.cloud import bigquery
from google.cloud import aiplatform
import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.preview.generative_models import GenerativeModel
from dotenv import load_dotenv

# Load environment
load_dotenv()

def check_env_vars():
    """Check required environment variables"""
    print("\n" + "="*60)
    print("1. CHECKING ENVIRONMENT VARIABLES")
    print("="*60)
    
    required = ["GCP_PROJECT_ID"]
    optional = ["BQ_DATASET", "GCP_REGION"]
    
    issues = []
    
    for var in required:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: NOT SET")
            issues.append(f"Missing required variable: {var}")
    
    for var in optional:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {value}")
        else:
            print(f"⚠️  {var}: NOT SET (will use default)")
    
    return issues

def check_bigquery_access(project_id):
    """Check BigQuery dataset access"""
    print("\n" + "="*60)
    print("2. CHECKING BIGQUERY ACCESS")
    print("="*60)
    
    issues = []
    
    try:
        # Initialize client WITHOUT location parameter
        client = bigquery.Client(project=project_id)
        print(f"✅ BigQuery client initialized for project: {project_id}")
        
        # List datasets
        datasets = list(client.list_datasets())
        
        if datasets:
            print(f"\n📦 Found {len(datasets)} dataset(s):")
            for dataset in datasets:
                dataset_ref = client.get_dataset(dataset.dataset_id)
                print(f"   - {dataset.dataset_id} (Location: {dataset_ref.location})")
        else:
            print("⚠️  No datasets found in project")
            issues.append("No BigQuery datasets found")
        
        # Check for 'conversational' dataset specifically
        dataset_id = os.getenv("BQ_DATASET", "conversational")
        try:
            dataset_ref = client.get_dataset(f"{project_id}.{dataset_id}")
            print(f"\n✅ Target dataset '{dataset_id}' found!")
            print(f"   Location: {dataset_ref.location}")
            print(f"   Created: {dataset_ref.created}")
            
            # List tables in dataset
            tables = list(client.list_tables(dataset_ref))
            if tables:
                print(f"\n📊 Tables in {dataset_id}:")
                for table in tables:
                    print(f"   - {table.table_id}")
            else:
                print(f"\n⚠️  No tables found in {dataset_id}")
                issues.append(f"Dataset {dataset_id} exists but has no tables")
                
        except Exception as e:
            print(f"\n❌ Dataset '{dataset_id}' not found: {e}")
            issues.append(f"Target dataset '{dataset_id}' does not exist")
        
    except Exception as e:
        print(f"❌ Error accessing BigQuery: {e}")
        issues.append(f"BigQuery access failed: {str(e)}")
    
    return issues

def check_vertex_ai_access(project_id):
    """Check Vertex AI model access"""
    print("\n" + "="*60)
    print("3. CHECKING VERTEX AI ACCESS")
    print("="*60)
    
    issues = []
    llm_region = "us-central1"
    
    try:
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=llm_region)
        aiplatform.init(project=project_id, location=llm_region)
        print(f"✅ Vertex AI initialized (Region: {llm_region})")
        
        # Test embedding model
        try:
            embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
            test_result = embedding_model.get_embeddings(["test"])
            print(f"✅ Embedding model accessible (Dimension: {len(test_result[0].values)})")
        except Exception as e:
            print(f"❌ Embedding model error: {e}")
            issues.append(f"Embedding model not accessible: {str(e)}")
        
        # Test Gemini model
        try:
            llm = GenerativeModel("gemini-1.5-pro-002")
            response = llm.generate_content("Say 'Hello'")
            print(f"✅ Gemini LLM accessible")
            print(f"   Test response: {response.text[:50]}...")
        except Exception as e:
            print(f"❌ Gemini LLM error: {e}")
            issues.append(f"Gemini LLM not accessible: {str(e)}")
            
            # Try flash model as fallback
            try:
                llm_flash = GenerativeModel("gemini-1.5-flash-002")
                response = llm_flash.generate_content("Say 'Hello'")
                print(f"ℹ️  Gemini Flash accessible as alternative")
            except:
                print(f"❌ Gemini Flash also not accessible")
        
    except Exception as e:
        print(f"❌ Error initializing Vertex AI: {e}")
        issues.append(f"Vertex AI initialization failed: {str(e)}")
    
    return issues

def check_table_schema(project_id, dataset_id):
    """Check if required tables exist with correct schema"""
    print("\n" + "="*60)
    print("4. CHECKING TABLE SCHEMAS")
    print("="*60)
    
    issues = []
    required_tables = [
        "unified_customer_view",
        "support_tickets_embedded",
        "product_catalog_embedded"
    ]
    
    try:
        client = bigquery.Client(project=project_id)
        
        for table_name in required_tables:
            try:
                table_ref = client.get_table(f"{project_id}.{dataset_id}.{table_name}")
                print(f"✅ {table_name}")
                print(f"   Rows: {table_ref.num_rows:,}")
                print(f"   Columns: {len(table_ref.schema)}")
                
                # Check for embedding column in embedded tables
                if "embedded" in table_name:
                    has_embedding = any(field.name == "embedding" for field in table_ref.schema)
                    if has_embedding:
                        print(f"   ✅ Has 'embedding' column")
                    else:
                        print(f"   ❌ Missing 'embedding' column")
                        issues.append(f"{table_name} missing embedding column")
                
            except Exception as e:
                print(f"❌ {table_name}: NOT FOUND")
                issues.append(f"Required table {table_name} not found")
        
    except Exception as e:
        print(f"❌ Error checking tables: {e}")
        issues.append(f"Table check failed: {str(e)}")
    
    return issues

def test_sample_query(project_id, dataset_id):
    """Test a sample query to verify end-to-end access"""
    print("\n" + "="*60)
    print("5. TESTING SAMPLE QUERY")
    print("="*60)
    
    issues = []
    
    try:
        client = bigquery.Client(project=project_id)
        
        # Test query on unified_customer_view
        query = f"""
        SELECT 
            COUNT(*) as customer_count,
            COUNT(DISTINCT customer_segment) as segments,
            COUNT(DISTINCT churn_risk) as risk_levels
        FROM `{project_id}.{dataset_id}.unified_customer_view`
        LIMIT 1
        """
        
        print(f"Running test query on unified_customer_view...")
        results = list(client.query(query).result())
        
        if results:
            row = results[0]
            print(f"✅ Query successful!")
            print(f"   Total customers: {row.customer_count:,}")
            print(f"   Segments: {row.segments}")
            print(f"   Risk levels: {row.risk_levels}")
        else:
            print(f"⚠️  Query returned no results")
            issues.append("Sample query returned empty results")
        
        # Test embedding query if table exists
        try:
            embed_query = f"""
            SELECT COUNT(*) as embedded_count
            FROM `{project_id}.{dataset_id}.support_tickets_embedded`
            WHERE embedding IS NOT NULL
            LIMIT 1
            """
            
            print(f"\nChecking embedded vectors...")
            embed_results = list(client.query(embed_query).result())
            
            if embed_results and embed_results[0].embedded_count > 0:
                print(f"✅ Found {embed_results[0].embedded_count:,} embedded documents")
            else:
                print(f"⚠️  No embedded documents found")
                issues.append("No embeddings found in support_tickets_embedded")
                
        except Exception as e:
            print(f"⚠️  Embedding check skipped: {e}")
        
    except Exception as e:
        print(f"❌ Sample query failed: {e}")
        issues.append(f"Sample query error: {str(e)}")
    
    return issues

def main():
    """Run all diagnostic checks"""
    print("\n" + "="*70)
    print("  CONVERSATIONAL 360 - ENVIRONMENT DIAGNOSTICS")
    print("="*70)
    
    all_issues = []
    
    # Check 1: Environment variables
    all_issues.extend(check_env_vars())
    
    project_id = os.getenv("GCP_PROJECT_ID")
    if not project_id:
        print("\n❌ CRITICAL: GCP_PROJECT_ID not set. Cannot continue.")
        sys.exit(1)
    
    # Check 2: BigQuery access
    all_issues.extend(check_bigquery_access(project_id))
    
    # Check 3: Vertex AI access
    all_issues.extend(check_vertex_ai_access(project_id))
    
    # Check 4: Table schemas
    dataset_id = os.getenv("BQ_DATASET", "conversational")
    all_issues.extend(check_table_schema(project_id, dataset_id))
    
    # Check 5: Sample queries
    all_issues.extend(test_sample_query(project_id, dataset_id))
    
    # Summary
    print("\n" + "="*70)
    print("  DIAGNOSTIC SUMMARY")
    print("="*70)
    
    if not all_issues:
        print("\n🎉 ALL CHECKS PASSED! Your environment is configured correctly.")
        print("\nYou can now run the RAG system:")
        print("   python src/rag_system.py")
        print("\nOr start the Streamlit app:")
        print("   streamlit run app.py")
    else:
        print(f"\n⚠️  Found {len(all_issues)} issue(s):\n")
        for i, issue in enumerate(all_issues, 1):
            print(f"{i}. {issue}")
        
        print("\n📋 RECOMMENDED FIXES:\n")
        
        if any("dataset" in issue.lower() for issue in all_issues):
            print("• Dataset Issues:")
            print("  - Ensure Fivetran has synced data to BigQuery")
            print("  - Run: python scripts/setup_schema.py")
            print("  - Verify dataset location matches BQ setup (usually 'US')")
        
        if any("table" in issue.lower() for issue in all_issues):
            print("\n• Table Issues:")
            print("  - Check if SQL views were created successfully")
            print("  - Run: python scripts/setup_schema.py")
            print("  - Verify Fivetran connectors are active")
        
        if any("embedding" in issue.lower() for issue in all_issues):
            print("\n• Embedding Issues:")
            print("  - Run: python scripts/generate_embeddings.py")
            print("  - This will generate vector embeddings for all documents")
        
        if any("vertex" in issue.lower() or "gemini" in issue.lower()):
            print("\n• Vertex AI Issues:")
            print("  - Enable Vertex AI API in GCP Console")
            print("  - Ensure you have proper IAM permissions")
            print("  - Try region 'us-central1' for Gemini models")
        
        print("\n" + "="*70)
        sys.exit(1)

if __name__ == "__main__":
    main()