#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from google.cloud import bigquery
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
DATASET_ID = "conversational"
SQL_DIR = Path(__file__).parent.parent / "sql"

def execute_sql_file(client, file_path):
    print(f"\n📝 Executing {file_path.name}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        sql = f.read()
    
    statements = [s.strip() for s in sql.split(';') if s.strip()]
    
    for i, statement in enumerate(statements, 1):
        if not statement:
            continue
            
        try:
            query_job = client.query(statement)
            query_job.result()
            print(f"   ✅ Statement {i}/{len(statements)} completed")
        except Exception as e:
            error_msg = str(e)
            # Only show critical errors, skip vector index errors (expected before embeddings)
            if "empty in the training input table" in error_msg:
                print(f"   ⚠️  Skipped statement {i} - will work after embeddings are generated")
            else:
                print(f"   ❌ Error on statement {i}: {error_msg[:200]}")

def main():
    if not PROJECT_ID:
        print("❌ Error: GCP_PROJECT_ID not set in .env file")
        sys.exit(1)
    
    print("="*60)
    print("Setting up BigQuery Schema")
    print("="*60)
    print(f"Project: {PROJECT_ID}")
    print(f"Dataset: {DATASET_ID}")
    print("="*60)
    
    client = bigquery.Client(project=PROJECT_ID)
    
    # Process SQL files in order
    sql_files = [
        SQL_DIR / "01_create_customer_360_view.sql",
        SQL_DIR / "02_create_embeddings_tables.sql",
        SQL_DIR / "04_analytics_views.sql",
        # Skip 03 for now - will run after embeddings
    ]
    
    for sql_file in sql_files:
        if sql_file.exists():
            execute_sql_file(client, sql_file)
        else:
            print(f"⚠️  File not found: {sql_file}")
    
    print("\n" + "="*60)
    print("✅ Schema setup complete!")
    print("Next step: python scripts/generate_embeddings.py --limit 100")
    print("="*60)

if __name__ == "__main__":
    main()