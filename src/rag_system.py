"""
Enterprise Customer 360 RAG System - FINAL PRODUCTION VERSION
Complete system with all fixes and defensive error handling
"""

from google.cloud import bigquery
from google.cloud import aiplatform
from vertexai.preview.generative_models import (
    GenerativeModel, Content, Part, Tool, FunctionDeclaration,
    GenerationConfig
)
from vertexai.language_models import TextEmbeddingModel 

import vertexai
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime
import json
import time
import re
from enum import Enum

from src.data_models import (
    CustomerContext, SearchResult, 
    ProductRecommendation, CustomerSegment, ChurnRisk, SourceType
)

# Placeholder utilities - Replace with your actual implementations
def format_currency(value): 
    """Format currency values"""
    return f"${value:,.2f}"

def calculate_health_score(context): 
    """Calculate customer health score based on multiple factors"""
    score = 100
    
    # Deduct points for churn risk
    if hasattr(context.churn_risk, 'value'):
        churn_value = context.churn_risk.value
    else:
        churn_value = str(context.churn_risk)
    
    if 'HIGH' in churn_value.upper():
        score -= 30
    elif 'MEDIUM' in churn_value.upper():
        score -= 15
    
    # Deduct points for inactivity
    if context.days_since_last_purchase:
        if context.days_since_last_purchase > 90:
            score -= 20
        elif context.days_since_last_purchase > 60:
            score -= 10
    
    # Deduct points for open tickets
    if context.open_tickets > 3:
        score -= 15
    elif context.open_tickets > 0:
        score -= 5
    
    # Add points for high lifetime value
    if context.lifetime_value > 5000:
        score += 10
    
    return max(0, min(100, score))

def sanitize_text(text): 
    """Sanitize text for embedding"""
    if not text:
        return ""
    return str(text).strip()


class QueryIntent(str, Enum):
    """Types of user query intents"""
    CUSTOMER_INFO = "customer_info"
    SUPPORT_HISTORY = "support_history"
    PRODUCT_RECOMMENDATION = "product_recommendation"
    CHURN_ANALYSIS = "churn_analysis"
    GENERAL = "general"


def classify_query(query: str, customer_available: bool) -> QueryIntent:
    """
    Classifies the user's natural language query into a specific business intent
    to determine the optimal retrieval strategy.
    """
    query_lower = query.lower()
    
    # Check for product recommendation intent
    if any(word in query_lower for word in ["product", "recommend", "suggest", "buy", "purchase"]):
        return QueryIntent.PRODUCT_RECOMMENDATION
    
    # Check for support history intent
    if any(word in query_lower for word in ["ticket", "support", "issue", "problem", "complaint"]):
        return QueryIntent.SUPPORT_HISTORY
    
    # Check for churn analysis intent
    if any(word in query_lower for word in ["churn", "risk", "leave", "retention", "cancel"]):
        return QueryIntent.CHURN_ANALYSIS
    
    # Check for customer info intent
    if customer_available and any(word in query_lower for word in ["who", "what", "profile", "information", "detail"]):
        return QueryIntent.CUSTOMER_INFO
    
    return QueryIntent.GENERAL


CLASSIFICATION_TOOL = Tool(
    function_declarations=[
        FunctionDeclaration(
            name="classify_query",
            description=classify_query.__doc__.strip(),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The user's question"},
                    "customer_available": {"type": "boolean", "description": "Boolean indicating if customer exists"}
                },
                "required": ["query", "customer_available"],
            },
        ),
    ]
)


class Customer360RAGSystem:
    """
    Production-grade RAG system for Customer 360 intelligence
    
    Features:
    - Semantic search across support tickets and product catalogs
    - AI-powered query classification and intent detection
    - Context-aware response generation with citations
    - Defensive error handling throughout
    - Support for multiple email domains (.com, .net, .org, etc.)
    """
    
    def __init__(
        self,
        project_id: str,
        dataset_id: str = "conversational",
        location: str = "US",
        embedding_model_name: str = "text-embedding-004",
        llm_model_name: str = "gemini-2.5-flash"
    ):
        """
        Initialize the RAG system
        
        Args:
            project_id: GCP project ID
            dataset_id: BigQuery dataset name
            location: BigQuery dataset location (US, EU, etc.)
            embedding_model_name: Vertex AI embedding model
            llm_model_name: Vertex AI LLM model
        """
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.bq_location = location
        self.llm_region = "us-central1"  # Vertex AI region
        
        print(f"🔧 Initializing RAG System...")
        print(f"   Project: {project_id}")
        print(f"   BigQuery Dataset: {dataset_id} (Location: {self.bq_location})")
        print(f"   Vertex AI Region: {self.llm_region}")
        
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=self.llm_region)
        aiplatform.init(project=project_id, location=self.llm_region)
        
        # Initialize BigQuery client (no location param - allows cross-region queries)
        self.bq_client = bigquery.Client(project=project_id)
        
        # Initialize embedding model
        try:
            self.embedding_model = TextEmbeddingModel.from_pretrained(embedding_model_name)
            self.embedding_dimension = 768  # text-embedding-004 dimension
            print(f"   ✅ Embedding model loaded: {embedding_model_name}")
        except Exception as e:
            print(f"   ⚠️  Warning loading embedding model: {e}")
            self.embedding_model = None
            self.embedding_dimension = 768
        
        # Initialize LLM
        try:
            self.llm_model_name = llm_model_name
            self.llm = GenerativeModel(llm_model_name)
            print(f"   ✅ LLM loaded: {llm_model_name}")
        except Exception as e:
            print(f"   ⚠️  Warning loading LLM: {e}")
            self.llm = None
        
        # Configuration
        self.max_search_results = 5
        self.temperature = 0.2
        self.max_output_tokens = 2048
        
        print(f"✅ RAG System initialized successfully")

    def get_all_customers(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get list of all customers with basic info
        Useful for populating dropdowns or search suggestions
        
        Args:
            limit: Maximum number of customers to return
            
        Returns:
            List of customer dictionaries with email, name, segment
        """
        query = f"""
        SELECT 
            customer_id,
            email, 
            first_name, 
            last_name,
            customer_segment,
            churn_risk,
            lifetime_value
        FROM `{self.project_id}.{self.dataset_id}.unified_customer_view`
        ORDER BY lifetime_value DESC
        LIMIT {limit}
        """
        
        try:
            results = self.bq_client.query(query).result()
            
            customers = []
            for row in results:
                name = f"{row.first_name or ''} {row.last_name or ''}".strip()
                customers.append({
                    "customer_id": row.customer_id,
                    "email": row.email,
                    "name": name,
                    "segment": row.customer_segment,
                    "churn_risk": row.churn_risk,
                    "lifetime_value": float(row.lifetime_value or 0)
                })
            
            return customers
            
        except Exception as e:
            print(f"❌ Error fetching customer list: {e}")
            return []

    def find_similar_customers(self, email_query: str, limit: int = 5) -> List[Dict[str, str]]:
        """
        Find customers with similar email addresses
        Useful for handling typos or suggesting alternatives
        
        Args:
            email_query: The email to search for
            limit: Maximum number of suggestions
            
        Returns:
            List of similar customers with edit distance scores
        """
        # Extract username from email (part before @)
        username = email_query.split('@')[0] if '@' in email_query else email_query
        
        query = f"""
        SELECT 
            email, 
            first_name, 
            last_name,
            customer_segment,
            EDIT_DISTANCE(LOWER(email), LOWER(@search_email)) as distance
        FROM `{self.project_id}.{self.dataset_id}.unified_customer_view`
        WHERE LOWER(email) LIKE LOWER(CONCAT('%', @username, '%'))
        ORDER BY distance ASC
        LIMIT {limit}
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("search_email", "STRING", email_query),
                bigquery.ScalarQueryParameter("username", "STRING", username)
            ]
        )
        
        try:
            results = self.bq_client.query(query, job_config=job_config).result()
            
            similar_customers = []
            for row in results:
                name = f"{row.first_name or ''} {row.last_name or ''}".strip()
                similar_customers.append({
                    "email": row.email,
                    "name": name,
                    "segment": row.customer_segment,
                    "distance": row.distance
                })
            
            return similar_customers
            
        except Exception as e:
            print(f"❌ Error finding similar customers: {e}")
            return []

    def get_customer_context(self, customer_email: str) -> Optional[CustomerContext]:
        """
        Fetch complete customer 360 context from BigQuery
        Handles all email domains (.com, .net, .org, etc.)
        
        Args:
            customer_email: Customer email address (case-insensitive)
            
        Returns:
            CustomerContext object or None if not found
        """
        query = f"""
        SELECT 
            customer_id, email, first_name, last_name, phone,
            total_orders, lifetime_value, days_since_last_purchase, 
            avg_order_value, 
            total_tickets, open_tickets, 
            avg_resolution_hours, avg_satisfaction_score, customer_segment, churn_risk,
            last_purchase_date, last_visit, total_sessions, total_page_views, visited_pages,
            customer_since
        FROM `{self.project_id}.{self.dataset_id}.unified_customer_view`
        WHERE LOWER(email) = LOWER(@email)
        LIMIT 1
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("email", "STRING", customer_email)
            ]
        )
        
        try:
            results = list(self.bq_client.query(query, job_config=job_config).result())
            
            if not results:
                print(f"ℹ️  No customer found with email: {customer_email}")
                
                # Try to suggest similar emails
                similar = self.find_similar_customers(customer_email, limit=3)
                if similar:
                    print(f"💡 Did you mean one of these?")
                    for s in similar:
                        print(f"   - {s['email']} ({s['name']})")
                
                return None
            
            row = results[0]
            
            # DEFENSIVE: Handle customer_segment enum conversion
            segment_value = row.customer_segment
            if segment_value:
                # Try direct enum value match
                if segment_value in [s.value for s in CustomerSegment]:
                    segment = CustomerSegment(segment_value)
                # Try uppercase enum name match
                elif segment_value.upper() in CustomerSegment.__members__:
                    segment = CustomerSegment[segment_value.upper()]
                # Try with underscore replacement
                elif segment_value.upper().replace(' ', '_') in CustomerSegment.__members__:
                    segment = CustomerSegment[segment_value.upper().replace(' ', '_')]
                else:
                    print(f"⚠️  Unknown segment '{segment_value}', defaulting to GENERAL")
                    segment = CustomerSegment.GENERAL
            else:
                segment = CustomerSegment.GENERAL
            
            # DEFENSIVE: Handle churn_risk enum conversion
            churn_risk_value = row.churn_risk
            if churn_risk_value:
                # Try direct enum value match
                if churn_risk_value in [c.value for c in ChurnRisk]:
                    churn_risk = ChurnRisk(churn_risk_value)
                # Try uppercase enum name match
                elif churn_risk_value.upper().replace(' ', '_') in ChurnRisk.__members__:
                    churn_risk = ChurnRisk[churn_risk_value.upper().replace(' ', '_')]
                else:
                    print(f"⚠️  Unknown churn risk '{churn_risk_value}', defaulting to LOW")
                    churn_risk = ChurnRisk.LOW
            else:
                churn_risk = ChurnRisk.LOW
            
            return CustomerContext(
                customer_id=row.customer_id,
                email=row.email,
                first_name=row.first_name or "",
                last_name=row.last_name or "",
                phone=row.phone,
                segment=segment,
                churn_risk=churn_risk,
                lifetime_value=float(row.lifetime_value or 0),
                total_orders=int(row.total_orders or 0),
                avg_order_value=float(row.avg_order_value or 0),
                days_since_last_purchase=row.days_since_last_purchase,
                total_sessions=int(row.total_sessions or 0),
                total_page_views=int(row.total_page_views or 0),
                open_tickets=int(row.open_tickets or 0),
                total_tickets=int(row.total_tickets or 0),
                avg_resolution_hours=row.avg_resolution_hours,
                satisfaction_score=row.avg_satisfaction_score,
                customer_since=row.customer_since,
                last_purchase_date=row.last_purchase_date,
                last_visit=row.last_visit,
                purchased_product_ids=[], 
                visited_pages=row.visited_pages or []
            )
            
        except Exception as e:
            print(f"❌ Error fetching customer context: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding vector for search query
        
        Args:
            query: Text query to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        if not self.embedding_model:
            print("⚠️  Embedding model not available, returning zero vector")
            return [0.0] * self.embedding_dimension
            
        try:
            sanitized_query = sanitize_text(query)
            if not sanitized_query:
                return [0.0] * self.embedding_dimension
                
            embedding_result = self.embedding_model.get_embeddings([sanitized_query])
            return embedding_result[0].values
            
        except Exception as e:
            print(f"❌ Error generating query embedding: {e}")
            return [0.0] * self.embedding_dimension

    def semantic_search_tickets(
        self,
        query: str,
        customer_email: Optional[str] = None,
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Search support tickets using semantic similarity
        
        Args:
            query: Search query
            customer_email: Optional filter by customer email
            top_k: Number of results to return
            
        Returns:
            List of SearchResult objects sorted by relevance
        """
        query_embedding = self._generate_query_embedding(query)
        
        # Build customer filter if provided
        customer_filter = ""
        if customer_email:
            customer_filter = "AND LOWER(customer_email) = LOWER(@customer_email)"
        
        search_query = f"""
        SELECT 
            ticket_id, customer_email, subject, full_text, status, priority, created_at, 
            (1 - COSINE_DISTANCE(embedding, @query_embedding)) AS similarity_score
        FROM `{self.project_id}.{self.dataset_id}.support_tickets_embedded`
        WHERE embedding IS NOT NULL
        {customer_filter}
        ORDER BY COSINE_DISTANCE(embedding, @query_embedding)
        LIMIT {top_k}
        """
        
        query_params = [
            bigquery.ArrayQueryParameter("query_embedding", "FLOAT64", query_embedding)
        ]
        
        if customer_email:
            query_params.append(
                bigquery.ScalarQueryParameter("customer_email", "STRING", customer_email)
            )
        
        job_config = bigquery.QueryJobConfig(query_parameters=query_params)
        
        try:
            results = self.bq_client.query(search_query, job_config=job_config).result()
            
            search_results = []
            for row in results:
                search_results.append(SearchResult(
                    doc_id=str(row.ticket_id),
                    content=row.full_text[:1000] if row.full_text else "",
                    source_type=SourceType.SUPPORT_TICKET,
                    similarity_score=float(row.similarity_score),
                    title=row.subject,
                    created_at=row.created_at,
                    metadata={
                        "status": row.status, 
                        "priority": row.priority,
                        "customer_email": row.customer_email
                    }
                ))
            
            return search_results
            
        except Exception as e:
            print(f"❌ Error in semantic search (tickets): {e}")
            import traceback
            traceback.print_exc()
            return []

    def semantic_search_products(
        self,
        query: str,
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Search product catalog using semantic similarity
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of SearchResult objects sorted by relevance
        """
        query_embedding = self._generate_query_embedding(query)
        
        search_query = f"""
        SELECT 
            product_id, product_name, description, category, price, full_text,
            (1 - COSINE_DISTANCE(embedding, @query_embedding)) AS similarity_score
        FROM `{self.project_id}.{self.dataset_id}.product_catalog_embedded`
        WHERE embedding IS NOT NULL
        ORDER BY COSINE_DISTANCE(embedding, @query_embedding)
        LIMIT {top_k}
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("query_embedding", "FLOAT64", query_embedding)
            ]
        )
        
        try:
            results = self.bq_client.query(search_query, job_config=job_config).result()
            
            search_results = []
            for row in results:
                search_results.append(SearchResult(
                    doc_id=str(row.product_id),
                    content=row.full_text[:500] if row.full_text else "",
                    source_type=SourceType.PRODUCT,
                    similarity_score=float(row.similarity_score),
                    title=row.product_name,
                    metadata={
                        "product_name": row.product_name, 
                        "category": row.category, 
                        "price": float(row.price) if row.price else 0.0,
                        "description": row.description
                    }
                ))
            
            return search_results
            
        except Exception as e:
            print(f"❌ Error in semantic search (products): {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _classify_query_intent(
        self, 
        query: str, 
        customer_context: Optional[CustomerContext] = None
    ) -> QueryIntent:
        """
        Classify user query intent using Gemini function calling
        Falls back to rule-based classification if LLM unavailable
        
        Args:
            query: User query text
            customer_context: Optional customer context
            
        Returns:
            QueryIntent enum value
        """
        if not self.llm:
            print("⚠️  LLM not available, using fallback classification")
            return classify_query(query, customer_context is not None)
            
        customer_available = customer_context is not None
        
        prompt = (
            f"Analyze the following user query: '{query}'. "
            f"Determine the primary intent. A customer profile is {'available' if customer_available else 'NOT available'}. "
            "Use the 'classify_query' tool."
        )
        
        try:
            response = self.llm.generate_content(
                prompt,
                tools=[CLASSIFICATION_TOOL],
                generation_config=GenerationConfig(temperature=0.0)
            )
            
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call'):
                        call = part.function_call
                        if call.name == "classify_query":
                            return classify_query(
                                call.args.get('query', query), 
                                call.args.get('customer_available', customer_available)
                            )

        except Exception as e:
            print(f"⚠️  LLM classification failed: {e}. Using fallback.")

        return classify_query(query, customer_available)

    def _retrieve_context(
        self,
        query: str,
        customer_email: Optional[str] = None,
        intent: Optional[QueryIntent] = None
    ) -> Tuple[Optional[CustomerContext], List[SearchResult]]:
        """
        Orchestrate retrieval from multiple data sources based on query intent
        
        Args:
            query: User query
            customer_email: Optional customer email for context
            intent: Optional pre-classified intent
            
        Returns:
            Tuple of (CustomerContext, List[SearchResult])
        """
        # Get customer context if email provided
        customer_context = None
        if customer_email:
            customer_context = self.get_customer_context(customer_email)
        
        # Classify intent if not provided
        if intent is None:
            intent = self._classify_query_intent(query, customer_context)
        
        print(f"🎯 Query intent classified as: {intent.value}")
        
        # Retrieve relevant documents based on intent
        all_results = []
        
        if intent == QueryIntent.SUPPORT_HISTORY:
            # Focus on support tickets
            ticket_results = self.semantic_search_tickets(query, customer_email, top_k=5)
            all_results.extend(ticket_results)
        
        elif intent == QueryIntent.PRODUCT_RECOMMENDATION:
            # Focus on products
            product_results = self.semantic_search_products(query, top_k=5)
            all_results.extend(product_results)
        
        elif intent == QueryIntent.CHURN_ANALYSIS:
            # Get recent support tickets for churn analysis
            ticket_results = self.semantic_search_tickets(query, customer_email, top_k=3)
            all_results.extend(ticket_results)
        
        else:  # CUSTOMER_INFO or GENERAL
            # Get mixed results
            ticket_results = self.semantic_search_tickets(query, customer_email, top_k=3)
            product_results = self.semantic_search_products(query, top_k=2)
            all_results.extend(ticket_results)
            all_results.extend(product_results)
        
        # Sort by similarity and filter low-quality results
        all_results.sort(key=lambda x: x.similarity_score, reverse=True)
        all_results = [r for r in all_results if r.similarity_score > 0.7][:self.max_search_results]
        
        print(f"📚 Retrieved {len(all_results)} relevant documents")
        
        return customer_context, all_results
    
    def _build_rag_prompt(
        self,
        query: str,
        customer_context: Optional[CustomerContext],
        search_results: List[SearchResult]
    ) -> str:
        """
        Build context-rich prompt for LLM generation
        
        Args:
            query: User query
            customer_context: Optional customer context
            search_results: Retrieved search results
            
        Returns:
            Complete prompt string
        """
        prompt_parts = []
        
        # System instructions
        prompt_parts.append("""You are an AI customer intelligence assistant for enterprise customer service.

Your role:
- Provide helpful, accurate insights about customers based on their data.
- Always ground your responses in the provided context.
- Cite your sources using [Source N] notation.
- If you don't have enough information, say so clearly.

Guidelines:
- Only use information from the provided context.
- Never make up or hallucinate data.
- Be concise but comprehensive.
- Focus on actionable insights.
""")
        
        # Add customer profile if available
        if customer_context:
            health_score = calculate_health_score(customer_context)
            
            # DEFENSIVE: Handle enum values safely
            segment_value = (customer_context.segment.value 
                           if hasattr(customer_context.segment, 'value') 
                           else str(customer_context.segment))
            churn_risk_value = (customer_context.churn_risk.value 
                              if hasattr(customer_context.churn_risk, 'value') 
                              else str(customer_context.churn_risk))
            
            prompt_parts.append(f"""
CUSTOMER PROFILE:
- Name: {customer_context.name}
- Email: {customer_context.email}
- Customer ID: {customer_context.customer_id}
- Segment: {segment_value}
- Lifetime Value: {format_currency(customer_context.lifetime_value)}
- Total Orders: {customer_context.total_orders}
- Average Order Value: {format_currency(customer_context.avg_order_value)}
- Days Since Last Purchase: {customer_context.days_since_last_purchase}
- Total Support Tickets: {customer_context.total_tickets}
- Open Support Tickets: {customer_context.open_tickets}
- Satisfaction Score: {customer_context.satisfaction_score or 'N/A'}/5.0
- Churn Risk: {churn_risk_value}
- Health Score: {health_score}/100
- Customer Since: {customer_context.customer_since}
""")
        
        # Add retrieved documents
        if search_results:
            prompt_parts.append("\nRELEVANT HISTORICAL DATA:")
            for i, result in enumerate(search_results, 1):
                # DEFENSIVE: Handle enum values safely
                source_type_value = (result.source_type.value 
                                   if hasattr(result.source_type, 'value') 
                                   else str(result.source_type))
                source_type = source_type_value.replace('_', ' ').title()
                similarity = f"{result.similarity_score:.1%}"
                
                prompt_parts.append(f"\n[Source {i}] ({source_type}, Relevance: {similarity})")
                if result.title:
                    prompt_parts.append(f"Title: {result.title}")
                if result.metadata.get('created_at'):
                    prompt_parts.append(f"Date: {result.metadata['created_at']}")
                prompt_parts.append(f"Content: {result.content[:800]}")
                
        # Add user query
        prompt_parts.append(f"\nUSER QUERY: {query}")
        
        # Add final instructions
        prompt_parts.append("""
INSTRUCTIONS:
Provide a helpful, data-driven response to the user's query. Use the customer profile and historical data to inform your answer. Cite specific sources using [Source N] notation when referencing information.

If recommending actions, be specific and prioritize based on the customer's risk level and value.

RESPONSE:""")
        
        return "\n".join(prompt_parts)

    def _generate_response(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_schema: Optional[dict] = None
    ) -> str:
        """
        Generate response using Gemini LLM
        
        Args:
            prompt: Complete prompt
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            response_schema: Optional JSON schema for structured output
            
        Returns:
            Generated response text
        """
        if not self.llm:
            return "I apologize, but the LLM is not available at this time."
            
        try:
            # Build generation config
            gen_config = GenerationConfig(
                temperature=temperature or self.temperature,
                top_p=0.8,
                top_k=40,
                max_output_tokens=max_tokens or self.max_output_tokens,
            )
            
            # Add structured output schema if provided
            if response_schema:
                gen_config.response_mime_type = "application/json"
                gen_config.response_schema = response_schema
                
            response = self.llm.generate_content(
                prompt,
                generation_config=gen_config
            )
            
            return response.text
            
        except Exception as e:
            print(f"❌ Error generating LLM response: {e}")
            import traceback
            traceback.print_exc()
            return f"I apologize, but I encountered an error generating a response: {str(e)}"

    def _parse_citations(
        self, 
        response_text: str, 
        search_results: List[SearchResult]
    ) -> List[Dict]:
        """
        Extract citation references from response and link to source documents
        
        Args:
            response_text: Generated response containing [Source N] citations
            search_results: List of source documents
            
        Returns:
            List of citation dictionaries
        """
        citations = []
        
        try:
            # Find all [Source N] references
            pattern = r'\[Source (\d+)\]'
            matches = re.finditer(pattern, response_text)
            cited_indices = set()
            
            for match in matches:
                try:
                    index = int(match.group(1)) - 1
                    if 0 <= index < len(search_results):
                        cited_indices.add(index)
                except (ValueError, IndexError) as e:
                    print(f"⚠️ Warning: Invalid citation index: {e}")
                    continue
            
            # Build citation objects for each cited source
            for idx in sorted(cited_indices):
                try:
                    result = search_results[idx]
                    
                    # DEFENSIVE: Handle both Enum and string values
                    if hasattr(result.source_type, 'value'):
                        source_type_str = result.source_type.value
                    else:
                        source_type_str = str(result.source_type)
                    
                    citations.append({
                        "source_id": idx + 1,
                        "doc_id": str(result.doc_id),
                        "source_type": source_type_str,
                        "title": str(result.title) if result.title else "",
                        "content": str(result.content)[:200] if result.content else "",
                        "similarity_score": float(result.similarity_score) if result.similarity_score else 0.0,
                        "metadata": result.metadata if result.metadata else {}
                    })
                except Exception as e:
                    print(f"⚠️ Warning: Could not process citation {idx}: {e}")
                    continue
            
        except Exception as e:
            print(f"❌ Error in citation parsing: {e}")
        
        return citations

    def answer_query(
        self,
        query: str,
        customer_email: Optional[str] = None,
        include_context: bool = True,
        response_schema: Optional[dict] = None
    ) -> Dict[str, Any]:
        """
        Complete RAG pipeline: Retrieve relevant context and generate grounded response
        
        This is the main entry point for the RAG system. It orchestrates:
        1. Customer context retrieval
        2. Query intent classification
        3. Semantic search across data sources
        4. Prompt construction
        5. LLM response generation
        6. Citation extraction
        
        Args:
            query: User's natural language query
            customer_email: Optional customer email for personalized context
            include_context: Whether to include customer context in response
            response_schema: Optional JSON schema for structured output
            
        Returns:
            Dictionary containing:
            - query: Original query
            - answer: Generated response
            - citations: List of source citations
            - metadata: Query metadata (intent, timing, etc.)
            - customer_context: Optional customer summary
        """
        start_time = time.time()
        
        try:
            # Step 1: Get customer context for intent classification
            customer_context_for_intent = None
            if customer_email:
                try:
                    customer_context_for_intent = self.get_customer_context(customer_email)
                    if customer_context_for_intent:
                        print(f"✅ Loaded context for: {customer_context_for_intent.name}")
                except Exception as e:
                    print(f"⚠️ Warning: Could not load customer context: {e}")
            
            # Step 2: Classify query intent
            try:
                intent = self._classify_query_intent(query, customer_context_for_intent)
            except Exception as e:
                print(f"⚠️ Warning: Intent classification failed: {e}, using GENERAL")
                intent = QueryIntent.GENERAL
            
            # Step 3: Retrieve context (customer + search results)
            try:
                customer_context, search_results = self._retrieve_context(
                    query=query,
                    customer_email=customer_email,
                    intent=intent
                )
            except Exception as e:
                print(f"❌ Error retrieving context: {e}")
                customer_context = customer_context_for_intent
                search_results = []
            
            # Step 4: Build RAG prompt
            try:
                prompt = self._build_rag_prompt(
                    query=query,
                    customer_context=customer_context,
                    search_results=search_results
                )
            except Exception as e:
                print(f"❌ Error building prompt: {e}")
                prompt = f"Answer this query based on available information: {query}"
            
            # Step 5: Generate response
            try:
                answer = self._generate_response(prompt, response_schema=response_schema)
            except Exception as e:
                print(f"❌ Error generating response: {e}")
                answer = f"I apologize, but I encountered an error: {str(e)}"
            
            # Step 6: Parse citations (DEFENSIVE)
            citations = []
            try:
                if search_results and not response_schema:  # Skip citation parsing for structured output
                    citations = self._parse_citations(answer, search_results)
            except Exception as e:
                print(f"⚠️ Warning: Citation parsing failed: {e}")
                citations = []
            
            # Calculate response time
            response_time_ms = int((time.time() - start_time) * 1000)
            print(f"⏱️  Response generated in {response_time_ms}ms")
            
            # Build response dictionary
            response = {
                "query": query,
                "answer": answer,
                "citations": citations,
                "metadata": {
                    "intent": intent.value if hasattr(intent, 'value') else str(intent),
                    "num_sources": len(search_results),
                    "response_time_ms": response_time_ms,
                    "model_used": self.llm_model_name,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
            # Add customer context summary if available and requested
            if include_context and customer_context:
                try:
                    response["customer_context"] = {
                        "customer_id": customer_context.customer_id,
                        "name": customer_context.name,
                        "email": customer_context.email,
                        "segment": (customer_context.segment.value 
                                  if hasattr(customer_context.segment, 'value') 
                                  else str(customer_context.segment)),
                        "churn_risk": (customer_context.churn_risk.value 
                                     if hasattr(customer_context.churn_risk, 'value') 
                                     else str(customer_context.churn_risk)),
                        "lifetime_value": customer_context.lifetime_value,
                        "health_score": calculate_health_score(customer_context)
                    }
                except Exception as e:
                    print(f"⚠️ Warning: Could not add customer context to response: {e}")
            
            return response
            
        except Exception as e:
            # Ultimate fallback for catastrophic errors
            print(f"❌ CRITICAL ERROR in answer_query: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "query": query,
                "answer": f"I apologize, but I encountered a critical error: {str(e)}",
                "citations": [],
                "metadata": {
                    "intent": "error",
                    "num_sources": 0,
                    "response_time_ms": int((time.time() - start_time) * 1000),
                    "model_used": self.llm_model_name if hasattr(self, 'llm_model_name') else "unknown",
                    "timestamp": datetime.utcnow().isoformat(),
                    "error": str(e)
                }
            }

    # ============================================================
    # SPECIALIZED AGENT METHODS
    # ============================================================

    def analyze_churn_risk(self, customer_email: str) -> Dict[str, Any]:
        """
        Specialized agent for comprehensive churn risk analysis
        
        Args:
            customer_email: Customer email address
            
        Returns:
            Analysis with churn factors and retention recommendations
        """
        query = """Analyze this customer's churn risk in detail:
1. What are the key risk factors?
2. What is their engagement trend?
3. What specific actions should we take to retain them?
4. What is the urgency level?

Provide concrete, actionable recommendations."""
        
        return self.answer_query(query, customer_email=customer_email)
    
    def recommend_products(
        self, 
        customer_email: str, 
        context: str = "",
        num_recommendations: int = 5
    ) -> Dict[str, Any]:
        """
        Specialized agent for personalized product recommendations
        
        Args:
            customer_email: Customer email address
            context: Optional additional context (e.g., "for a birthday gift")
            num_recommendations: Number of products to recommend
            
        Returns:
            Structured product recommendations with reasoning
        """
        query = f"""Recommend {num_recommendations} products for this customer based on:
- Their purchase history and preferences
- Their customer segment and value
- Current trends in their category
{f"- Additional context: {context}" if context else ""}

For each recommendation, explain why it's a good fit.
Your final output MUST be a JSON array of recommendations."""
        
        # Define structured output schema
        product_item_schema = ProductRecommendation.model_json_schema()
        product_schema = {
            "type": "array",
            "items": product_item_schema
        }
        
        response = self.answer_query(
            query=query, 
            customer_email=customer_email, 
            include_context=True,
            response_schema=product_schema
        )
        
        # Parse structured recommendations
        try:
            recommendations = json.loads(response['answer'])
            response['recommendations'] = recommendations
            response['answer'] = f"Successfully generated {len(recommendations)} personalized product recommendations."
            response['citations'] = []  # Citations don't apply to structured output
        except json.JSONDecodeError as e:
            response['error'] = "Failed to parse structured JSON output."
            print(f"❌ JSON parsing error: {e}")
            
        return response
    
    def summarize_support_history(self, customer_email: str) -> Dict[str, Any]:
        """
        Specialized agent for support history summarization
        
        Args:
            customer_email: Customer email address
            
        Returns:
            Summary with recent tickets and recurring issues
        """
        query = """Summarize this customer's support history:
1. List their most recent support tickets with status
2. Identify any recurring themes or issues
3. Assess their overall satisfaction level
4. Recommend any proactive support actions

Be specific and reference actual ticket data."""
        
        return self.answer_query(query, customer_email=customer_email)
    
    def generate_customer_summary(self, customer_email: str) -> Dict[str, Any]:
        """
        Generate comprehensive customer summary for quick review
        
        Args:
            customer_email: Customer email address
            
        Returns:
            Executive summary of customer status and needs
        """
        query = """Provide a comprehensive executive summary of this customer:
1. Overall customer health and value
2. Recent activity and engagement
3. Key opportunities (upsell, cross-sell, retention)
4. Immediate action items
5. Risk factors to monitor

Keep it concise but actionable."""
        
        return self.answer_query(query, customer_email=customer_email)
    
    def compare_customers(
        self, 
        email1: str, 
        email2: str
    ) -> Dict[str, Any]:
        """
        Compare two customers for segmentation or analysis
        
        Args:
            email1: First customer email
            email2: Second customer email
            
        Returns:
            Comparative analysis
        """
        # Get both customer contexts
        customer1 = self.get_customer_context(email1)
        customer2 = self.get_customer_context(email2)
        
        if not customer1 or not customer2:
            return {
                "query": f"Compare {email1} and {email2}",
                "answer": "One or both customers could not be found.",
                "citations": [],
                "metadata": {"intent": "error"}
            }
        
        # Build comparison prompt
        query = f"""Compare these two customers and identify:
1. Key similarities and differences
2. Which customer is more valuable and why
3. Which customer has higher churn risk
4. Different engagement strategies needed for each

Customer 1: {customer1.name} ({email1})
Customer 2: {customer2.name} ({email2})"""
        
        # Note: This is a simple implementation. For production, you might want
        # to pass both contexts through the RAG pipeline more elegantly
        return self.answer_query(query, customer_email=email1)


# ============================================================
# UTILITY FUNCTIONS FOR EXTERNAL USE
# ============================================================

def create_rag_system(
    project_id: Optional[str] = None,
    dataset_id: str = "conversational",
    location: str = "US"
) -> Customer360RAGSystem:
    """
    Factory function to create a RAG system instance with error handling
    
    Args:
        project_id: GCP project ID (defaults to environment variable)
        dataset_id: BigQuery dataset name
        location: BigQuery dataset location
        
    Returns:
        Initialized Customer360RAGSystem instance
    """
    import os
    
    if not project_id:
        project_id = os.getenv("GCP_PROJECT_ID")
        if not project_id:
            raise ValueError("project_id must be provided or GCP_PROJECT_ID must be set")
    
    return Customer360RAGSystem(
        project_id=project_id,
        dataset_id=dataset_id,
        location=location
    )


if __name__ == "__main__":
    # Example usage and testing
    print("Customer 360 RAG System - Production Version")
    print("=" * 60)
    
    # This section is for testing only
    # In production, this module should be imported and used via Streamlit
    
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Initialize system
    rag = create_rag_system()
    
    # Test: Get all customers
    print("\n📋 Fetching customer list...")
    customers = rag.get_all_customers(limit=10)
    print(f"Found {len(customers)} customers")
    
    if customers:
        # Test: Get first customer details
        test_email = customers[0]["email"]
        print(f"\n🔍 Testing with customer: {test_email}")
        
        customer = rag.get_customer_context(test_email)
        if customer:
            print(f"✅ Customer loaded: {customer.name}")
            print(f"   Segment: {customer.segment.value if hasattr(customer.segment, 'value') else customer.segment}")
            print(f"   LTV: {format_currency(customer.lifetime_value)}")
            print(f"   Health Score: {calculate_health_score(customer)}/100")
            
            # Test: Answer a query
            print(f"\n💬 Testing query...")
            response = rag.answer_query(
                query="What is this customer's churn risk?",
                customer_email=test_email
            )
            print(f"✅ Response generated in {response['metadata']['response_time_ms']}ms")
            print(f"   Intent: {response['metadata']['intent']}")
            print(f"   Sources: {response['metadata']['num_sources']}")
            print(f"\nAnswer preview: {response['answer'][:200]}...")
        else:
            print("❌ Could not load customer")
    
    print("\n✅ All tests completed")