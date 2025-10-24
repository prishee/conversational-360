#!/usr/bin/env python3
"""
Generate embeddings for all documents in BigQuery - FINAL CORRECTED VERSION
"""

import argparse
import sys
import os
from pathlib import Path
import time
from typing import List, Dict
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from google.cloud import bigquery
from vertexai.language_models import TextEmbeddingModel
import vertexai
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate and populate embeddings in BigQuery"""
    
    def __init__(
        self,
        project_id: str,
        dataset_id: str,
        location: str = "us-central1",
        batch_size: int = 5
    ):
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.location = location
        self.batch_size = batch_size
        
        logger.info("Initializing Google Cloud clients...")
        vertexai.init(project=project_id, location=location)
        self.bq_client = bigquery.Client(project=project_id)
        self.embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
        
        logger.info("✅ Clients initialized successfully")
    
    def get_documents_without_embeddings(
        self,
        table_name: str,
        limit: int = None
    ) -> List[Dict]:
        """Fetch documents that don't have embeddings yet"""
        
        # Determine ID column based on table
        if 'support_tickets' in table_name:
            id_column = 'ticket_id'
        elif 'product' in table_name:
            id_column = 'product_id'
        else:
            id_column = 'doc_id'
        
        limit_clause = f"LIMIT {limit}" if limit else ""
        
        # FIXED: No unnecessary CAST, let BigQuery handle the type
        query = f"""
        SELECT 
            {id_column} as doc_id,
            full_text as content
        FROM `{self.project_id}.{self.dataset_id}.{table_name}`
        WHERE embedding IS NULL
          AND full_text IS NOT NULL
          AND LENGTH(full_text) > 10
        {limit_clause}
        """
        
        logger.info(f"Fetching documents from {table_name}...")
        query_job = self.bq_client.query(query)
        results = list(query_job.result())
        
        documents = [
            {"doc_id": str(row.doc_id), "content": row.content}
            for row in results
        ]
        
        logger.info(f"Found {len(documents)} documents without embeddings")
        return documents, id_column
    
    def generate_embeddings_batch(
        self,
        texts: List[str]
    ) -> List[List[float]]:
        """Generate embeddings for a batch of texts"""
        try:
            embeddings_result = self.embedding_model.get_embeddings(texts)
            embeddings = [emb.values for emb in embeddings_result]
            time.sleep(0.1)  # Rate limiting
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return [[0.0] * 768] * len(texts)
    
    def update_embeddings_in_bigquery(
        self,
        table_name: str,
        doc_ids: List[str],
        embeddings: List[List[float]],
        id_column: str
    ):
        """Update BigQuery with generated embeddings using MERGE"""
        
        # Build VALUES clause for MERGE
        values_parts = []
        for doc_id, embedding in zip(doc_ids, embeddings):
            embedding_str = "[" + ",".join(map(str, embedding)) + "]"
            values_parts.append(f"('{doc_id}', {embedding_str})")
        
        values_clause = ",\n".join(values_parts)
        
        # FIXED: Removed unnecessary CAST on target column
        merge_query = f"""
        MERGE INTO `{self.project_id}.{self.dataset_id}.{table_name}` T
        USING (
          SELECT {id_column}, embedding
          FROM UNNEST([
            STRUCT<{id_column} STRING, embedding ARRAY<FLOAT64>>
            {values_clause}
          ])
        ) S
        ON T.{id_column} = S.{id_column}
        WHEN MATCHED THEN
          UPDATE SET 
            T.embedding = S.embedding,
            T.embedding_model = 'text-embedding-004',
            T.embedded_at = CURRENT_TIMESTAMP()
        """
        
        try:
            self.bq_client.query(merge_query).result()
        except Exception as e:
            logger.error(f"Error updating batch: {e}")
    
    def process_table(
        self,
        table_name: str,
        limit: int = None
    ):
        """Process all documents in a table"""
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing table: {table_name}")
        logger.info(f"{'='*60}\n")
        
        documents, id_column = self.get_documents_without_embeddings(table_name, limit)
        
        if not documents:
            logger.info("✅ All documents already have embeddings!")
            return
        
        total_docs = len(documents)
        logger.info(f"Generating embeddings for {total_docs} documents...")
        
        success_count = 0
        error_count = 0
        
        with tqdm(total=total_docs, desc="Processing") as pbar:
            for i in range(0, total_docs, self.batch_size):
                batch_docs = documents[i:i + self.batch_size]
                
                doc_ids = [doc["doc_id"] for doc in batch_docs]
                texts = [doc["content"] for doc in batch_docs]
                
                try:
                    embeddings = self.generate_embeddings_batch(texts)
                    
                    self.update_embeddings_in_bigquery(
                        table_name,
                        doc_ids,
                        embeddings,
                        id_column
                    )
                    
                    success_count += len(batch_docs)
                    pbar.update(len(batch_docs))
                    
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                    error_count += len(batch_docs)
                    pbar.update(len(batch_docs))
        
        logger.info(f"\n✅ Completed!")
        logger.info(f"   Success: {success_count} documents")
        logger.info(f"   Errors: {error_count} documents")
    
    def create_vector_index(self, table_name: str):
        """Create vector index for fast similarity search"""
        
        logger.info(f"\nCreating vector index for {table_name}...")
        
        index_name = f"{table_name}_embedding_idx"
        
        index_query = f"""
        CREATE VECTOR INDEX IF NOT EXISTS {index_name}
        ON `{self.project_id}.{self.dataset_id}.{table_name}`(embedding)
        OPTIONS(
            distance_type = 'COSINE',
            index_type = 'IVF',
            ivf_options = '{{"num_lists": 100}}'
        )
        """
        
        try:
            self.bq_client.query(index_query).result()
            logger.info("✅ Vector index created successfully")
        except Exception as e:
            logger.warning(f"Index creation note: {e}")
    
    def get_embedding_stats(self, table_name: str) -> Dict:
        """Get statistics about embeddings in a table"""
        
        query = f"""
        SELECT 
            COUNT(*) as total_rows,
            COUNT(embedding) as rows_with_embeddings,
            COUNT(*) - COUNT(embedding) as rows_without_embeddings,
            ROUND(COUNT(embedding) / COUNT(*) * 100, 2) as completion_percentage
        FROM `{self.project_id}.{self.dataset_id}.{table_name}`
        """
        
        result = list(self.bq_client.query(query).result())[0]
        
        return {
            "total_rows": result.total_rows,
            "rows_with_embeddings": result.rows_with_embeddings,
            "rows_without_embeddings": result.rows_without_embeddings,
            "completion_percentage": result.completion_percentage
        }


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings for documents in BigQuery"
    )
    parser.add_argument(
        "--project-id",
        type=str,
        default=os.getenv("GCP_PROJECT_ID"),
        help="GCP Project ID"
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="conversational",
        help="BigQuery dataset ID"
    )
    parser.add_argument(
        "--location",
        type=str,
        default="us-central1",
        help="GCP region"
    )
    parser.add_argument(
        "--table",
        type=str,
        choices=["support_tickets_embedded", "product_catalog_embedded", "all"],
        default="all",
        help="Which table to process"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Batch size for embedding generation"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of documents to process (for testing)"
    )
    parser.add_argument(
        "--create-index",
        action="store_true",
        help="Create vector indexes after embedding generation"
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show statistics, don't generate embeddings"
    )
    
    args = parser.parse_args()
    
    if not args.project_id:
        logger.error("❌ Project ID is required. Set GCP_PROJECT_ID env var or use --project-id")
        sys.exit(1)
    
    try:
        generator = EmbeddingGenerator(
            project_id=args.project_id,
            dataset_id=args.dataset_id,
            location=args.location,
            batch_size=args.batch_size
        )
    except Exception as e:
        logger.error(f"❌ Failed to initialize: {e}")
        sys.exit(1)
    
    tables = []
    if args.table == "all":
        tables = ["support_tickets_embedded", "product_catalog_embedded"]
    else:
        tables = [args.table]
    
    if args.stats_only:
        logger.info("\n📊 Embedding Statistics:\n")
        for table in tables:
            try:
                stats = generator.get_embedding_stats(table)
                logger.info(f"{table}:")
                logger.info(f"  Total rows: {stats['total_rows']}")
                logger.info(f"  With embeddings: {stats['rows_with_embeddings']}")
                logger.info(f"  Without embeddings: {stats['rows_without_embeddings']}")
                logger.info(f"  Completion: {stats['completion_percentage']}%\n")
            except Exception as e:
                logger.error(f"Error getting stats for {table}: {e}\n")
        sys.exit(0)
    
    start_time = time.time()
    
    for table in tables:
        try:
            generator.process_table(table, limit=args.limit)
            
            if args.create_index:
                generator.create_vector_index(table)
                
        except Exception as e:
            logger.error(f"❌ Error processing {table}: {e}")
            continue
    
    elapsed_time = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"🎉 All processing complete!")
    logger.info(f"   Total time: {elapsed_time:.2f} seconds")
    logger.info(f"{'='*60}\n")
    
    logger.info("📊 Final Statistics:\n")
    for table in tables:
        try:
            stats = generator.get_embedding_stats(table)
            logger.info(f"{table}:")
            logger.info(f"  Completion: {stats['completion_percentage']}%")
            logger.info(f"  Remaining: {stats['rows_without_embeddings']} documents\n")
        except Exception as e:
            logger.error(f"Error getting final stats for {table}: {e}\n")


if __name__ == "__main__":
    main()