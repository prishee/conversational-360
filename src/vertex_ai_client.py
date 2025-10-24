"""
AI-Powered Customer Intelligence Agent
Uses Fivetran data in BigQuery with Vertex AI for RAG and agentic workflows
"""

from google.cloud import bigquery
from google.cloud import aiplatform
from vertexai.preview.generative_models import GenerativeModel, Part, Content, Tool
from vertexai.preview import rag
import vertexai
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import json
import time

# Define a placeholder for the RAG Corpus (in a real scenario, you'd manage this outside the class)
# We will use a list of Parts/Dicts to simulate the RAG context within the query_with_rag method

class CustomerIntelligenceAgent:
    """
    Client class for orchestrating BigQuery data retrieval and Vertex AI/Gemini generation.
    """
    def __init__(self, project_id: str, location: str = "us-central1"):
        """Initialize the agent with Google Cloud credentials"""
        self.project_id = project_id
        self.location = location
        
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=location)
        
        # Initialize BigQuery client
        self.bq_client = bigquery.Client(project=project_id)
        
        # Initialize Gemini model (Using 1.5 Pro for agentic logic and RAG)
        self.model = GenerativeModel("gemini-1.5-pro")
        
        # Placeholder for RAG Corpus/Resource Name (not used directly in this sample, but kept for structure)
        self.rag_resource_name = None 
        
        print(f" Customer Intelligence Agent initialized for project: {project_id}")
        
    def setup_rag_corpus(self, dataset_id: str, table_id: str) -> List[Dict[str, Any]]:
        """
        Fetches recent customer interactions from BigQuery and formats them 
        as a list of documents for use as RAG context (in-memory simulation).
        """
        # --- BigQuery Query (Completed) ---
        query = f"""
        SELECT 
            interaction_id,
            customer_id,
            timestamp,
            channel,
            sentiment_score,
            content,
            metadata
        FROM `{self.project_id}.{dataset_id}.{table_id}`
        WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
        ORDER BY timestamp DESC
        LIMIT 100 
        """
        
        # Execute query
        query_job = self.bq_client.query(query)
        results = query_job.result()
        
        # Convert to documents for RAG (list of dicts)
        documents = []
        for row in results:
            doc_text = f"""
            Interaction ID: {row.interaction_id}
            Customer: {row.customer_id}
            Channel: {row.channel}
            Sentiment: {row.sentiment_score}
            Date: {row.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
            Content: {row.content}
            """
            documents.append({
                "id": str(row.interaction_id),
                "content": doc_text.strip(),
                "metadata": dict(row.metadata) if row.metadata else {}
            })
        
        return documents

    def analyze_customer_sentiment(self, customer_id: str, dataset_id: str, table_id: str) -> dict:
        """
        Analyzes and summarizes sentiment trends for a specific customer from BigQuery.
        
        Args:
            customer_id: The ID of the customer.
        
        Returns:
            A dictionary containing sentiment analysis results.
        """
        # --- BigQuery Query (Completed) ---
        query = f"""
        SELECT 
            DATE(timestamp) as date,
            AVG(sentiment_score) as avg_sentiment,
            COUNT(*) as num_interactions
        FROM `{self.project_id}.{dataset_id}.{table_id}`
        WHERE customer_id = @customer_id
        AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
        GROUP BY 1
        ORDER BY date DESC
        LIMIT 30
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("customer_id", "STRING", customer_id)
            ]
        )
        
        try:
            results = list(self.bq_client.query(query, job_config=job_config).result())
            
            if not results:
                return {"customer_id": customer_id, "summary": "No recent interactions found for this customer."}
                
            sentiment_data = [
                {"date": row.date.strftime('%Y-%m-%d'), 
                 "avg_sentiment": round(row.avg_sentiment, 3), 
                 "interactions": row.num_interactions}
                for row in results
            ]
            
            # Use Gemini to generate a quick summary based on the results
            data_summary = json.dumps(sentiment_data)
            summary_prompt = (
                f"Analyze the following recent sentiment data for customer {customer_id}:\n\n"
                f"{data_summary}\n\n"
                f"Provide a concise summary of the sentiment trend (improving, declining, volatile) "
                f"and highlight the average sentiment score over the period. The scores range from -1.0 (negative) to 1.0 (positive)."
            )
            
            summary_response = self.model.generate_content(
                summary_prompt,
                config={"temperature": 0.1}
            ).text
            
            return {
                "customer_id": customer_id,
                "sentiment_summary": summary_response,
                "data": sentiment_data,
                "latest_avg_sentiment": sentiment_data[0]['avg_sentiment'] if sentiment_data else 0.0
            }

        except Exception as e:
            return {"customer_id": customer_id, "error": f"Error fetching or analyzing sentiment: {e}"}

    def query_with_rag(self, query: str, context_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generates an answer using the provided query and RAG context.
        Simulates RAG by injecting documents into the prompt (similar effect to Vertex AI's RAG).
        """
        
        # Format documents into a context block
        context_block = []
        for i, doc in enumerate(context_documents):
            context_block.append(f"--- SOURCE {i+1} ---\n{doc.get('content')}\n--- END SOURCE {i+1} ---\n")

        rag_prompt = f"""
        You are an expert customer intelligence analyst. Use ONLY the 'RELEVANT CONTEXT' provided below to answer the user's 'QUERY'. 
        Cite the source number (e.g., [Source 1], [Source 2]) for every piece of factual information you state.

        RELEVANT CONTEXT:
        {'\n'.join(context_block)}

        QUERY: {query}
        
        RESPONSE (Grounded in context, cite sources):
        """
        
        start_time = time.time()
        
        try:
            response = self.model.generate_content(
                rag_prompt,
                config={"temperature": 0.1}
            )
            
            response_time_ms = int((time.time() - start_time) * 1000)

            # Simple citation parsing (as done in the rag_system.py)
            citations = []
            for i, doc in enumerate(context_documents):
                if f"[Source {i+1}]" in response.text:
                    citations.append({
                        "source_id": i + 1,
                        "doc_id": doc.get('id'),
                        "content_snippet": doc.get('content')[:200] + "...",
                        "metadata": doc.get('metadata', {})
                    })
            
            return {
                "answer": response.text,
                "citations": citations,
                "response_time_ms": response_time_ms,
                "status": "success"
            }
        
        except Exception as e:
            return {"answer": f"Error during RAG generation: {e}", "citations": [], "status": "error"}

    def orchestrate_query(self, customer_id: str, query: str, dataset_id: str, table_id: str) -> Dict[str, Any]:
        """
        Agentic workflow: orchestrate data fetching, specialized analysis, and RAG generation.
        """
        print(f"-> Orchestrating query for customer {customer_id}: {query}")

        # Step 1: Gather specialized data (e.g., Sentiment Analysis)
        sentiment_analysis = self.analyze_customer_sentiment(customer_id, dataset_id, table_id)
        
        # Step 2: Retrieve recent raw interaction documents (RAG context)
        # Note: In a full system, this would be a filtered semantic search, 
        # but here we use the generic setup_rag_corpus filtered by customer_id if available.
        all_rag_documents = self.setup_rag_corpus(dataset_id, table_id)
        
        customer_rag_docs = [
            doc for doc in all_rag_documents if doc.get('content') and customer_id in doc.get('content')
        ]
        
        # Step 3: Combine all context into a single RAG query
        # We will summarize the sentiment data into a context document
        sentiment_doc = {
            "id": "SENTIMENT_SUMMARY",
            "content": f"Customer {customer_id} Sentiment Analysis:\nSummary: {sentiment_analysis.get('sentiment_summary')}\nLatest Average Score: {sentiment_analysis.get('latest_avg_sentiment')}",
            "metadata": {"type": "analysis"}
        }
        
        # Prioritize the sentiment summary, then the raw interactions
        combined_context = [sentiment_doc] + customer_rag_docs[:5] # Use up to 5 raw interactions
        
        # Step 4: Generate the final answer using RAG
        final_rag_response = self.query_with_rag(
            query=query, 
            context_documents=combined_context
        )
        
        return {
            "query": query,
            "customer_id": customer_id,
            "final_answer": final_rag_response['answer'],
            "rag_citations": final_rag_response['citations'],
            "sentiment_report": sentiment_analysis,
            "metadata": {
                "rag_sources_used": len(combined_context),
                "total_response_time_ms": final_rag_response['response_time_ms']
            }
        }

if __name__ == '__main__':
    # --- Example Usage (requires GCP setup and BigQuery tables) ---
    
    #  IMPORTANT: Replace these placeholders with your actual GCP details and table names
    YOUR_PROJECT_ID = "your-gcp-project-id"
    YOUR_DATASET_ID = "customer_360"
    YOUR_INTERACTION_TABLE = "customer_interactions"
    TEST_CUSTOMER_ID = "CUST-12345" # Replace with a known customer ID

    # To run this example, you must have a BigQuery table named
    # `your-gcp-project-id.customer_360.customer_interactions` with the columns:
    # interaction_id, customer_id, timestamp, channel, sentiment_score, content, metadata
    
    # agent = CustomerIntelligenceAgent(project_id=YOUR_PROJECT_ID)

    # test_query = "What is the recent trend in this customer's satisfaction and what were their last two interaction points about?"
    
    # final_result = agent.orchestrate_query(
    #     customer_id=TEST_CUSTOMER_ID,
    #     query=test_query,
    #     dataset_id=YOUR_DATASET_ID,
    #     table_id=YOUR_INTERACTION_TABLE
    # )

    # print("\n--- FINAL ORCHESTRATION RESULT ---")
    # print(json.dumps(final_result, indent=2))
    
    pass