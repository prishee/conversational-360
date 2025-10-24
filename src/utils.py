"""
Utility functions for Conversational 360
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import numpy as np
from src.data_models import CustomerContext, CustomerHealthScore


def format_currency(amount: float) -> str:
    """Format number as currency"""
    return f"${amount:,.2f}"


def format_date(date: Optional[datetime], format_str: str = "%Y-%m-%d") -> str:
    """Format datetime to string"""
    if date is None:
        return "N/A"
    return date.strftime(format_str)


def format_relative_time(date: Optional[datetime]) -> str:
    """Format date as relative time (e.g., '3 days ago')"""
    if date is None:
        return "Never"
    
    now = datetime.utcnow()
    delta = now - date
    
    if delta.days > 365:
        years = delta.days // 365
        return f"{years} year{'s' if years > 1 else ''} ago"
    elif delta.days > 30:
        months = delta.days // 30
        return f"{months} month{'s' if months > 1 else ''} ago"
    elif delta.days > 0:
        return f"{delta.days} day{'s' if delta.days > 1 else ''} ago"
    elif delta.seconds > 3600:
        hours = delta.seconds // 3600
        return f"{hours} hour{'s' if hours > 1 else ''} ago"
    elif delta.seconds > 60:
        minutes = delta.seconds // 60
        return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
    else:
        return "Just now"


def calculate_health_score(customer: CustomerContext) -> int:
    """
    Calculate overall customer health score (0-100)
    Based on multiple factors: recency, frequency, monetary value, satisfaction
    """
    scores = []
    
    # Recency score (0-100)
    if customer.days_since_last_purchase is not None:
        if customer.days_since_last_purchase < 30:
            recency = 100
        elif customer.days_since_last_purchase < 90:
            recency = 75
        elif customer.days_since_last_purchase < 180:
            recency = 50
        else:
            recency = 25
        scores.append(recency)
    
    # Frequency score (0-100)
    if customer.total_orders > 0:
        frequency = min(100, (customer.total_orders / 10.0) * 100)
        scores.append(frequency)
    
    # Monetary score (0-100)
    if customer.lifetime_value > 0:
        monetary = min(100, (customer.lifetime_value / 100.0))
        scores.append(monetary)
    
    # Satisfaction score (0-100)
    if customer.satisfaction_score:
        satisfaction = (customer.satisfaction_score / 5.0) * 100
        scores.append(satisfaction)
    
    # Support health (inverse of open tickets)
    if customer.open_tickets == 0:
        support = 100
    elif customer.open_tickets <= 2:
        support = 70
    else:
        support = 30
    scores.append(support)
    
    # Calculate weighted average
    if scores:
        return int(np.mean(scores))
    return 50  # Default neutral score


def calculate_detailed_health_score(customer: CustomerContext) -> CustomerHealthScore:
    """Calculate detailed health score with component breakdown"""
    
    # Recency score
    if customer.days_since_last_purchase is None:
        recency_score = 50
    elif customer.days_since_last_purchase < 30:
        recency_score = 100
    elif customer.days_since_last_purchase < 90:
        recency_score = 75
    elif customer.days_since_last_purchase < 180:
        recency_score = 50
    else:
        recency_score = 25
    
    # Frequency score
    frequency_score = min(100, int((customer.total_orders / 10.0) * 100))
    
    # Monetary score
    monetary_score = min(100, int(customer.lifetime_value / 100.0))
    
    # Satisfaction score
    if customer.satisfaction_score:
        satisfaction_score = int((customer.satisfaction_score / 5.0) * 100)
    else:
        satisfaction_score = 50
    
    # Support health score
    if customer.open_tickets == 0:
        support_health_score = 100
    elif customer.open_tickets <= 2:
        support_health_score = 70
    else:
        support_health_score = 30
    
    # Overall score (weighted average)
    overall_score = int(
        (recency_score * 0.25) +
        (frequency_score * 0.20) +
        (monetary_score * 0.25) +
        (satisfaction_score * 0.15) +
        (support_health_score * 0.15)
    )
    
    # Identify risk factors
    risk_factors = []
    if customer.days_since_last_purchase and customer.days_since_last_purchase > 90:
        risk_factors.append("No purchases in 90+ days")
    if customer.open_tickets > 2:
        risk_factors.append("Multiple open support tickets")
    if customer.satisfaction_score and customer.satisfaction_score < 3.5:
        risk_factors.append("Low satisfaction score")
    if customer.churn_risk.value == "High Risk":
        risk_factors.append("High churn risk identified")
    
    # Identify positive factors
    positive_factors = []
    if customer.lifetime_value > 10000:
        positive_factors.append("High lifetime value")
    if customer.total_orders > 20:
        positive_factors.append("Frequent purchaser")
    if customer.satisfaction_score and customer.satisfaction_score >= 4.5:
        positive_factors.append("Highly satisfied customer")
    if customer.segment.value == "VIP":
        positive_factors.append("VIP customer segment")
    
    # Determine trend (simplified - would compare to historical data in production)
    if overall_score >= 80:
        trend = "improving"
    elif overall_score < 60:
        trend = "declining"
    else:
        trend = "stable"
    
    return CustomerHealthScore(
        customer_id=customer.customer_id,
        overall_score=overall_score,
        recency_score=recency_score,
        frequency_score=frequency_score,
        monetary_score=monetary_score,
        satisfaction_score=satisfaction_score,
        support_health_score=support_health_score,
        trend=trend,
        risk_factors=risk_factors,
        positive_factors=positive_factors
    )


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    vec1_array = np.array(vec1)
    vec2_array = np.array(vec2)
    
    dot_product = np.dot(vec1_array, vec2_array)
    norm1 = np.linalg.norm(vec1_array)
    norm2 = np.linalg.norm(vec2_array)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def chunk_text(text: str, max_tokens: int = 8000, overlap: int = 200) -> List[str]:
    """
    Split text into chunks for processing
    Simple token estimation: ~4 chars per token
    """
    max_chars = max_tokens * 4
    overlap_chars = overlap * 4
    
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_chars
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for period, question mark, or exclamation within last 200 chars
            search_start = max(start, end - 200)
            last_period = text.rfind('.', search_start, end)
            last_question = text.rfind('?', search_start, end)
            last_exclamation = text.rfind('!', search_start, end)
            
            break_point = max(last_period, last_question, last_exclamation)
            if break_point > start:
                end = break_point + 1
        
        chunks.append(text[start:end])
        start = end - overlap_chars
    
    return chunks


def sanitize_text(text: str) -> str:
    """Clean and sanitize text for embedding/LLM processing"""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Remove null bytes
    text = text.replace('\x00', '')
    
    # Truncate if too long (safety check)
    max_length = 100000
    if len(text) > max_length:
        text = text[:max_length] + "..."
    
    return text.strip()


def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """
    Simple keyword extraction using frequency
    (In production, use more sophisticated methods like TF-IDF)
    """
    # Remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these',
        'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your'
    }
    
    # Tokenize and count
    words = text.lower().split()
    word_freq = {}
    
    for word in words:
        # Remove punctuation
        word = word.strip('.,!?;:()"\'')
        if len(word) > 3 and word not in stop_words:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top N
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_words[:top_n]]


def validate_email(email: str) -> bool:
    """Basic email validation"""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    return re.match(pattern, email) is not None


def calculate_churn_probability(customer: CustomerContext) -> float:
    """
    Calculate churn probability (0.0 - 1.0)
    Simple heuristic - in production, use ML model
    """
    score = 0.0
    
    # Days since purchase factor
    if customer.days_since_last_purchase:
        if customer.days_since_last_purchase > 180:
            score += 0.4
        elif customer.days_since_last_purchase > 90:
            score += 0.2
    
    # Open tickets factor
    if customer.open_tickets > 3:
        score += 0.3
    elif customer.open_tickets > 1:
        score += 0.15
    
    # Satisfaction factor
    if customer.satisfaction_score:
        if customer.satisfaction_score < 3.0:
            score += 0.3
        elif customer.satisfaction_score < 3.5:
            score += 0.15
    
    return min(1.0, score)


def generate_prompt_template(
    query: str,
    customer_context: Optional[CustomerContext],
    search_results: List[Dict[str, Any]],
    max_context_length: int = 8000
) -> str:
    """Generate prompt for LLM with customer context and search results"""
    
    prompt_parts = []
    
    # System context
    prompt_parts.append("""You are an AI customer intelligence assistant. Your role is to provide helpful, accurate, and personalized insights about customers based on their data.

Guidelines:
- Always ground your responses in the provided data
- Be empathetic and customer-focused
- Provide specific, actionable recommendations
- Cite your sources when making claims
- If you don't have enough information, say so
""")
    
    # Customer context
    if customer_context:
        prompt_parts.append(f"""
Customer Context:
- Name: {customer_context.name}
- Email: {customer_context.email}
- Segment: {customer_context.segment}
- Lifetime Value: ${customer_context.lifetime_value:,.2f}
- Total Orders: {customer_context.total_orders}
- Days Since Last Purchase: {customer_context.days_since_last_purchase or 'N/A'}
- Open Support Tickets: {customer_context.open_tickets}
- Satisfaction Score: {customer_context.satisfaction_score or 'N/A'}
- Churn Risk: {customer_context.churn_risk}
""")
    
    # Search results (relevant documents)
    if search_results:
        prompt_parts.append("\nRelevant Historical Data:")
        for i, result in enumerate(search_results[:5], 1):
            content = result.get('content', '')[:500]  # Truncate long content
            source_type = result.get('source_type', 'unknown')
            similarity = result.get('similarity_score', 0)
            
            prompt_parts.append(f"\n[Source {i}] ({source_type}, relevance: {similarity:.2%})")
            prompt_parts.append(content)
    
    # User query
    prompt_parts.append(f"\nUser Query: {query}")
    
    prompt_parts.append("\nProvide a helpful, data-driven response:")
    
    # Combine and truncate if needed
    full_prompt = "\n".join(prompt_parts)
    
    if len(full_prompt) > max_context_length * 4:  # Rough char-to-token estimate
        # Truncate search results if prompt is too long
        full_prompt = full_prompt[:max_context_length * 4]
    
    return full_prompt


def parse_citations_from_response(response_text: str, search_results: List[Dict]) -> List[Dict]:
    """
    Extract citations from LLM response
    Simple implementation - looks for [Source N] references
    """
    import re
    citations = []
    
    # Find all [Source N] references
    pattern = r'\[Source (\d+)\]'
    matches = re.finditer(pattern, response_text)
    
    cited_indices = set()
    for match in matches:
        index = int(match.group(1)) - 1  # Convert to 0-based
        if 0 <= index < len(search_results):
            cited_indices.add(index)
    
    # Return the cited search results
    for idx in cited_indices:
        citations.append(search_results[idx])
    
    return citations


def estimate_token_count(text: str) -> int:
    """Estimate token count (rough approximation: 1 token ≈ 4 characters)"""
    return len(text) // 4


def truncate_to_token_limit(text: str, max_tokens: int = 8000) -> str:
    """Truncate text to fit within token limit"""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def batch_items(items: List[Any], batch_size: int) -> List[List[Any]]:
    """Split list into batches"""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division that returns default value on division by zero"""
    try:
        return numerator / denominator if denominator != 0 else default
    except (TypeError, ZeroDivisionError):
        return default


def calculate_percentile(value: float, values: List[float]) -> float:
    """Calculate percentile rank of a value in a list"""
    if not values:
        return 0.0
    
    sorted_values = sorted(values)
    rank = sum(1 for v in sorted_values if v <= value)
    return (rank / len(sorted_values)) * 100