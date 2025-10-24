"""
Unit tests for RAG system
"""

import pytest
from unittest.mock import patch, Mock, MagicMock
from google.cloud import bigquery
from vertexai.language_models import TextEmbeddingModel
from vertexai.preview.generative_models import GenerativeModel
from src.rag_system import Customer360RAGSystem
from src.data_models import CustomerContext, Ticket

# --- MOCK DATA ---

MOCK_CUSTOMER_CONTEXT = {
    "id": "CUST-100",
    "name": "Alice Smith",
    "lifetime_value": 5000.00,
    "last_login": "2025-10-20",
    "subscription_tier": "Premium"
}

MOCK_TICKET_DATA = [
    bigquery.Row(("TICKET-001", "Billing issue", "2025-10-15", 0.9), ["id", "subject", "date", "sentiment"]),
    bigquery.Row(("TICKET-002", "Feature request for new dashboard", "2025-10-10", 0.2), ["id", "subject", "date", "sentiment"]),
]

MOCK_TICKET_EMBEDDINGS = [
    [0.1, 0.2, 0.3], # Embedding for TICKET-001 content
    [0.4, 0.5, 0.6], # Embedding for TICKET-002 content
]

MOCK_QUERY_EMBEDDING = [0.15, 0.25, 0.35]

# --- FIXTURES ---

# Mock the BigQuery Client initialization globally
@patch('google.cloud.bigquery.Client')
# Mock the TextEmbeddingModel initialization globally
@patch('vertexai.language_models.TextEmbeddingModel.from_pretrained')
# Mock the GenerativeModel initialization globally
@patch('vertexai.preview.generative_models.GenerativeModel')
@pytest.fixture
def rag_system(MockGenerativeModel, MockEmbeddingModel, MockBigQueryClient):
    """
    Create RAG system for testing with mocked external services.
    The mocks are passed as arguments to the fixture.
    """
    # 1. Mock BigQuery Client behavior
    mock_bq_client = MockBigQueryClient.return_value
    
    # Mock the query.result() for general customer context lookup
    mock_result_context = Mock()
    mock_result_context.one.return_value = bigquery.Row(tuple(MOCK_CUSTOMER_CONTEXT.values()), tuple(MOCK_CUSTOMER_CONTEXT.keys()))
    
    # Mock the query.result() for fetching all tickets (pre-search)
    mock_result_tickets = Mock()
    mock_result_tickets.result.return_value = MOCK_TICKET_DATA
    
    # Map specific query jobs to their mocked results
    mock_bq_client.query.side_effect = [
        # First query: get_customer_context (returns one row result)
        MagicMock(result=lambda: mock_result_context),
        # Second query: fetch_tickets (returns ticket data rows)
        MagicMock(result=lambda: MOCK_TICKET_DATA),
    ]

    # 2. Mock Embedding Model behavior
    mock_embedding_model = MockEmbeddingModel.return_value
    
    # Mock the get_embeddings to return a fixed vector for the query and content
    # In a real test, you'd map input text to output vector for more realism.
    # Here we simplify:
    def mock_get_embeddings(texts, **kwargs):
        # Simplistic mock: assumes the first text is the query and the rest are documents
        if len(texts) == 1 and "What is the customer's main concern?" in texts[0]:
            # Returns the embedding for the query
            mock_embedding = Mock()
            mock_embedding.values = MOCK_QUERY_EMBEDDING
            return [mock_embedding]
        
        # Returns embeddings for the ticket content
        mock_embeddings = []
        for i in range(len(texts)):
            mock_embedding = Mock()
            mock_embedding.values = MOCK_TICKET_EMBEDDINGS[i]
            mock_embeddings.append(mock_embedding)
        return mock_embeddings

    mock_embedding_model.get_embeddings.side_effect = mock_get_embeddings

    # 3. Mock Generative Model behavior (for final RAG answer)
    mock_gemini_model = MockGenerativeModel.return_value
    mock_response = Mock()
    mock_response.text = "The customer's primary concern is a **recent billing issue** [Source 1]."
    mock_gemini_model.generate_content.return_value = mock_response

    # Initialize the system, which triggers the mocked BigQuery/VertexAI initializations
    system = Customer360RAGSystem(
        project_id="test-project",
        dataset_id="test_dataset"
    )
    
    # Store the mocks on the system for easy assertion later
    system._mock_bq_client = mock_bq_client
    system._mock_embedding_model = mock_embedding_model
    system._mock_gemini_model = mock_gemini_model
    
    return system

# --- TESTS ---

def test_initialization_and_client_setup(rag_system):
    """Test that clients are initialized correctly"""
    assert rag_system.project_id == "test-project"
    # Check that BigQuery client was instantiated
    assert rag_system.bq_client is not None
    # Check that BigQuery client was called with the correct project
    rag_system._mock_bq_client.assert_called_once_with(project="test-project")
    # Check that the Embedding Model was loaded
    rag_system._mock_embedding_model.from_pretrained.assert_called_once()
    # Check that the Generative Model was loaded
    rag_system._mock_gemini_model.assert_called_once()

# ----------------------------------------------------------------------

def test_customer_360_retrieval(rag_system):
    """Test customer data retrieval and mapping to data model"""
    customer_id = "CUST-100"
    
    context = rag_system.get_customer_context(customer_id)
    
    # 1. Assert the correct BigQuery call was made
    rag_system._mock_bq_client.query.call_args[0][0].strip().startswith("SELECT")
    
    # 2. Assert the result is correctly mapped to the Pydantic model
    assert isinstance(context, CustomerContext)
    assert context.id == customer_id
    assert context.name == MOCK_CUSTOMER_CONTEXT["name"]
    assert context.subscription_tier == MOCK_CUSTOMER_CONTEXT["subscription_tier"]
    assert context.lifetime_value == MOCK_CUSTOMER_CONTEXT["lifetime_value"]

# ----------------------------------------------------------------------

def test_fetch_tickets(rag_system):
    """Test fetching raw ticket data and mapping to data models"""
    tickets = rag_system.fetch_tickets("CUST-100")
    
    # 1. Assert the correct number of tickets were returned
    assert len(tickets) == len(MOCK_TICKET_DATA)
    
    # 2. Assert the first ticket is correctly mapped
    first_ticket = tickets[0]
    assert isinstance(first_ticket, Ticket)
    assert first_ticket.id == "TICKET-001"
    assert first_ticket.subject == "Billing issue"
    # Check that content/embedding are correctly initialized (even if embedding is None initially)
    assert first_ticket.content is not None
    assert first_ticket.embedding is None

# --------------------------------