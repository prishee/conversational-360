"""
Data models for Conversational 360
Type-safe data structures using Pydantic
"""

from pydantic import BaseModel, EmailStr, Field, validator
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class ChurnRisk(str, Enum):
    """Customer churn risk levels"""
    HIGH = "High Risk"
    MEDIUM = "Medium Risk"
    LOW = "Low Risk"


class CustomerSegment(str, Enum):
    """Customer value segments"""
    VIP = "VIP"
    HIGH_VALUE = "High Value"
    MEDIUM_VALUE = "Medium Value"
    LOW_VALUE = "Low Value"
    GENERAL = "General" # Added for robustness in the RAG system


class SourceType(str, Enum):
    """Data source types"""
    SUPPORT_TICKET = "support_ticket"
    PRODUCT = "product"
    ORDER = "order"
    INTERACTION = "interaction"


class CustomerContext(BaseModel):
    """Complete customer 360° context"""
    customer_id: str
    email: EmailStr
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone: Optional[str] = None
    
    # Derived fields
    name: str = ""
    
    # Segmentation
    segment: CustomerSegment
    churn_risk: ChurnRisk
    
    # Financial metrics
    lifetime_value: float = Field(ge=0)
    total_orders: int = Field(ge=0)
    avg_order_value: float = Field(ge=0)
    
    # Engagement metrics
    days_since_last_purchase: Optional[int] = None
    total_sessions: int = Field(ge=0, default=0)
    total_page_views: int = Field(ge=0, default=0)
    
    # Support metrics
    open_tickets: int = Field(ge=0, default=0)
    total_tickets: int = Field(ge=0, default=0)
    avg_resolution_hours: Optional[float] = None
    satisfaction_score: Optional[float] = Field(ge=0, le=5, default=None)
    
    # Dates
    customer_since: Optional[datetime] = None
    last_purchase_date: Optional[datetime] = None
    last_visit: Optional[datetime] = None
    
    # Additional data
    purchased_product_ids: List[str] = []
    visited_pages: List[str] = []
    metadata: Dict[str, Any] = {}
    
    @validator('name', always=True)
    def set_name(cls, v, values):
        """Auto-generate full name from first and last name"""
        if not v:
            first = values.get('first_name', '')
            last = values.get('last_name', '')
            return f"{first} {last}".strip() or "Unknown"
        return v
    
    class Config:
        use_enum_values = True


class SearchResult(BaseModel):
    """Single search result from vector search"""
    doc_id: str
    content: str
    source_type: SourceType
    similarity_score: float = Field(ge=0, le=1)
    metadata: Dict[str, Any] = {}
    
    # Optional fields based on source type
    title: Optional[str] = None
    created_at: Optional[datetime] = None
    
    class Config:
        use_enum_values = True


class RAGResponse(BaseModel):
    """Response from RAG system"""
    query: str
    answer: str
    citations: List[Dict[str, Any]]
    customer_context: Optional[CustomerContext] = None
    confidence_score: float = Field(ge=0, le=1, default=0.8)
    response_time_ms: int
    
    # Metadata
    model_used: str
    tokens_used: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Optional field for specialized structured outputs (like product recommendations)
    recommendations: Optional[List[Dict[str, Any]]] = None

    # FIX: Suppress Pydantic UserWarning for 'model_used'
    class Config:
        protected_namespaces = ()


class ProductRecommendation(BaseModel):
    """AI-generated product recommendation (Pydantic structure for forced JSON output)"""
    product_id: str
    product_name: str
    description: str
    price: float
    category: str
    relevance_score: float = Field(ge=0, le=1)
    reason: str
    image_url: Optional[str] = None


class ActionRecommendation(BaseModel):
    """Recommended action for customer engagement"""
    action_type: str  # e.g., "retention_offer", "upsell", "support_escalation"
    priority: str  # "high", "medium", "low"
    title: str
    description: str
    expected_impact: str
    effort_required: str  # "low", "medium", "high"
    estimated_value: Optional[float] = None


class CustomerHealthScore(BaseModel):
    """Detailed customer health scoring"""
    customer_id: str
    overall_score: int = Field(ge=0, le=100)
    
    # Component scores
    recency_score: int = Field(ge=0, le=100)
    frequency_score: int = Field(ge=0, le=100)
    monetary_score: int = Field(ge=0, le=100)
    satisfaction_score: int = Field(ge=0, le=100)
    support_health_score: int = Field(ge=0, le=100)
    
    # Trend
    score_change: int = 0  # Change from last calculation
    trend: str = "stable"  # "improving", "declining", "stable"
    
    # Risk factors
    risk_factors: List[str] = []
    positive_factors: List[str] = []
    
    calculated_at: datetime = Field(default_factory=datetime.utcnow)


class EmbeddingRequest(BaseModel):
    """Request for embedding generation"""
    texts: List[str]
    model: str = "text-embedding-004"
    batch_size: int = Field(ge=1, le=10, default=5)

    # FIX: Suppress Pydantic UserWarning for 'model'
    class Config:
        protected_namespaces = ()


class EmbeddingResponse(BaseModel):
    """Response from embedding generation"""
    embeddings: List[List[float]]
    dimension: int = 768
    model_used: str
    processing_time_ms: int
    
    # FIX: Suppress Pydantic UserWarning for 'model_used'
    class Config:
        protected_namespaces = ()


class SystemMetrics(BaseModel):
    """System health and performance metrics"""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # API health
    api_status: str = "healthy"
    uptime_percentage: float = Field(ge=0, le=100)
    
    # Performance
    avg_query_latency_ms: int
    p95_query_latency_ms: int
    error_rate_percentage: float = Field(ge=0, le=100)
    
    # Usage
    total_queries_24h: int
    total_embeddings_24h: int
    bigquery_bytes_processed_24h: int
    
    # Costs
    bigquery_cost_24h: float
    vertex_ai_cost_24h: float
    total_cost_24h: float