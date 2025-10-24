"""
Script to regenerate embeddings in BigQuery with text-embedding-004 (768 dimensions)
Run this to fix the dimension mismatch issue
"""

from google.cloud import bigquery
from vertexai.language_models import TextEmbeddingModel
import vertexai
from typing import List
import time

def regenerate_embeddings(
    project_id: str = "conversational-360",
    dataset_id: str = "conversational",
    location: str = "us-central1"
):
    """Regenerate all embeddings with text-embedding-004"""
    
    # Initialize
    vertexai.init(project=project_id, location=location)
    bq_client = bigquery.Client(project=project_id)
    embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    
    print(f"🚀 Starting embedding regeneration...")
    print(f"   Model: text-embedding-004 (768 dimensions)")
    
    # ============================================================
    # REGENERATE SUPPORT TICKET EMBEDDINGS
    # ============================================================
    
    print(f"\n📋 Processing support_tickets_embedded...")
    
    # Fetch tickets without embeddings or with wrong dimension
    fetch_tickets_query = f"""
    SELECT ticket_id, full_text
    FROM `{project_id}.{dataset_id}.support_tickets_embedded`
    WHERE full_text IS NOT NULL
    """
    
    tickets = list(bq_client.query(fetch_tickets_query).result())
    print(f"   Found {len(tickets)} tickets to process")
    
    # Process in batches of 5 (API limit for embeddings)
    batch_size = 5
    updated_tickets = 0
    
    for i in range(0, len(tickets), batch_size):
        batch = tickets[i:i+batch_size]
        ticket_ids = [row.ticket_id for row in batch]
        texts = [row.full_text[:5000] for row in batch]  # Truncate to 5000 chars
        
        try:
            # Generate embeddings
            embeddings_result = embedding_model.get_embeddings(texts)
            embeddings = [emb.values for emb in embeddings_result]
            
            # Update each ticket
            for ticket_id, embedding in zip(ticket_ids, embeddings):
                update_query = f"""
                UPDATE `{project_id}.{dataset_id}.support_tickets_embedded`
                SET embedding = {embedding}
                WHERE ticket_id = '{ticket_id}'
                """
                bq_client.query(update_query).result()
                updated_tickets += 1
            
            print(f"   ✅ Processed {updated_tickets}/{len(tickets)} tickets")
            time.sleep(0.1)  # Rate limiting
            
        except Exception as e:
            print(f"   ❌ Error processing batch: {e}")
            continue
    
    print(f"   🎉 Completed! Updated {updated_tickets} tickets")
    
    # ============================================================
    # REGENERATE PRODUCT EMBEDDINGS
    # ============================================================
    
    print(f"\n🛍️  Processing product_catalog_embedded...")
    
    fetch_products_query = f"""
    SELECT product_id, full_text
    FROM `{project_id}.{dataset_id}.product_catalog_embedded`
    WHERE full_text IS NOT NULL
    """
    
    products = list(bq_client.query(fetch_products_query).result())
    print(f"   Found {len(products)} products to process")
    
    updated_products = 0
    
    for i in range(0, len(products), batch_size):
        batch = products[i:i+batch_size]
        product_ids = [row.product_id for row in batch]
        texts = [row.full_text[:5000] for row in batch]
        
        try:
            embeddings_result = embedding_model.get_embeddings(texts)
            embeddings = [emb.values for emb in embeddings_result]
            
            for product_id, embedding in zip(product_ids, embeddings):
                update_query = f"""
                UPDATE `{project_id}.{dataset_id}.product_catalog_embedded`
                SET embedding = {embedding}
                WHERE product_id = '{product_id}'
                """
                bq_client.query(update_query).result()
                updated_products += 1
            
            print(f"   ✅ Processed {updated_products}/{len(products)} products")
            time.sleep(0.1)
            
        except Exception as e:
            print(f"   ❌ Error processing batch: {e}")
            continue
    
    print(f"   🎉 Completed! Updated {updated_products} products")
    
    # ============================================================
    # VERIFY DIMENSIONS
    # ============================================================
    
    print(f"\n🔍 Verifying embedding dimensions...")
    
    verify_query = f"""
    SELECT 
      'support_tickets' as table_name,
      ARRAY_LENGTH(embedding) as dimension,
      COUNT(*) as count
    FROM `{project_id}.{dataset_id}.support_tickets_embedded`
    WHERE embedding IS NOT NULL
    GROUP BY dimension
    
    UNION ALL
    
    SELECT 
      'product_catalog' as table_name,
      ARRAY_LENGTH(embedding) as dimension,
      COUNT(*) as count
    FROM `{project_id}.{dataset_id}.product_catalog_embedded`
    WHERE embedding IS NOT NULL
    GROUP BY dimension
    ORDER BY table_name, dimension
    """
    
    results = list(bq_client.query(verify_query).result())
    
    print("\n📊 Verification Results:")
    for row in results:
        status = "✅" if row.dimension == 768 else "❌"
        print(f"   {status} {row.table_name}: {row.dimension} dimensions ({row.count} records)")
    
    print("\n✅ Embedding regeneration complete!")
    print("   You can now restart your Streamlit app.")

if __name__ == "__main__":
    regenerate_embeddings()