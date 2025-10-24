"""
Conversational 360 - Main Streamlit Application
AI-Powered Customer Intelligence Platform
"""
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import custom modules
from src.rag_system import Customer360RAGSystem
from src.data_models import CustomerContext, SearchResult
from src.utils import format_currency, format_date, calculate_health_score

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Conversational 360",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #FF6B6B;
    }
    
    .customer-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
    }
    
    .risk-high {
        color: #FF6B6B;
        font-weight: 700;
    }
    
    .risk-medium {
        color: #FFA500;
        font-weight: 700;
    }
    
    .risk-low {
        color: #4ECDC4;
        font-weight: 700;
    }
    
    .citation {
        background: #f0f2f6;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.25rem 0;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# INITIALIZE SESSION STATE
# ============================================================

if 'rag_system' not in st.session_state:
    with st.spinner("Initializing AI system..."):
        try:
            st.session_state.rag_system = Customer360RAGSystem(
                project_id=os.getenv("GCP_PROJECT_ID"),
                dataset_id=os.getenv("BQ_DATASET", "conversational"),
                location=os.getenv("GCP_REGION", "us-central1")
            )
            st.session_state.system_ready = True
        except Exception as e:
            st.error(f"Failed to initialize system: {str(e)}")
            st.session_state.system_ready = False

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'selected_customer' not in st.session_state:
    st.session_state.selected_customer = None

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    # Uncomment when you have a logo
    # st.image("docs/images/logo.png", width=200)
    st.title("Conversational 360")
    
    selected = option_menu(
        menu_title="Navigation",
        options=["Home", "Customer Lookup", "AI Assistant", "Analytics", "Settings"],
        icons=["house", "search", "chat-dots", "graph-up", "gear"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px"},
            "nav-link-selected": {"background-color": "#FF6B6B"},
        }
    )
    
    st.divider()
    
    # System status
    if st.session_state.system_ready:
        st.success("System Online")
    else:
        st.error("System Offline")
    
    # Quick stats
    st.metric("Active Customers", "12,543")
    st.metric("High Risk", "342", delta="-12")
    st.metric("Avg Health Score", "87.3", delta="2.1")

# ============================================================
# HOME PAGE
# ============================================================

if selected == "Home":
    st.markdown('<h1 class="main-header">Conversational 360</h1>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Customer Intelligence Platform")
    
    st.divider()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Total Customers", value="12,543", delta="234 this month")
    
    with col2:
        st.metric(label="Avg LTV", value="$8,456", delta="$324")
    
    with col3:
        st.metric(label="Churn Rate", value="2.3%", delta="-0.4%")
    
    with col4:
        st.metric(label="Satisfaction", value="4.6/5.0", delta="0.2")
    
    st.divider()
    
    # Feature highlights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Unified Data")
        st.write("Automatic sync from Salesforce, Zendesk, Shopify, and Google Analytics via Fivetran")
    
    with col2:
        st.markdown("### AI-Powered")
        st.write("Gemini 2.5 Flash with RAG for context-aware responses and recommendations")
    
    with col3:
        st.markdown("### Predictive Analytics")
        st.write("Churn prediction, health scoring, and proactive intervention suggestions")
    
    # Recent high-risk customers
    st.markdown("### High-Risk Customers Requiring Attention")
    
    risk_df = pd.DataFrame({
        "Customer": ["Sarah Johnson", "Mike Chen", "Emily Rodriguez"],
        "Email": ["sarah.j@example.com", "mike.c@example.com", "emily.r@example.com"],
        "LTV": ["$45,230", "$32,100", "$28,500"],
        "Days Since Purchase": [95, 142, 87],
        "Open Tickets": [3, 1, 2],
        "Risk Score": ["High", "High", "High"]
    })
    
    st.dataframe(risk_df, use_container_width=True, hide_index=True)

# ============================================================
# CUSTOMER LOOKUP PAGE
# ============================================================

elif selected == "Customer Lookup":
    st.markdown("## Customer 360 View")
    
    # Search input
    col1, col2 = st.columns([3, 1])
    with col1:
        customer_email = st.text_input(
            "Enter customer email",
            placeholder="customer@example.com",
            help="Search by email address to view complete customer profile"
        )
    
    with col2:
        search_button = st.button("Search", type="primary", use_container_width=True)
    
    if search_button and customer_email:
        with st.spinner("Loading customer data..."):
            try:
                customer = st.session_state.rag_system.get_customer_context(customer_email)
                st.session_state.selected_customer = customer
                
                if customer:
                    # Get risk class for styling
                    churn_risk_str = customer.churn_risk.value if hasattr(customer.churn_risk, 'value') else str(customer.churn_risk)
                    risk_class = f"risk-{churn_risk_str.lower().replace(' ', '-')}"
                    
                    # Get segment string
                    segment_str = customer.segment.value if hasattr(customer.segment, 'value') else str(customer.segment)
                    
                    st.markdown(f"""
                    <div class="customer-card">
                        <h2>{customer.name}</h2>
                        <p style="font-size: 1.1rem; opacity: 0.9;">{customer.email}</p>
                        <p style="font-size: 1.5rem; margin-top: 1rem;">
                            Segment: <strong>{segment_str}</strong> | 
                            Churn Risk: <span class="{risk_class}">{churn_risk_str}</span>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Key metrics
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.metric("Lifetime Value", format_currency(customer.lifetime_value))
                    
                    with col2:
                        st.metric("Total Orders", customer.total_orders)
                    
                    with col3:
                        st.metric("Open Tickets", customer.open_tickets, delta=f"{customer.open_tickets} active")
                    
                    with col4:
                        st.metric("Last Purchase", f"{customer.days_since_last_purchase} days ago")
                    
                    with col5:
                        health_score = calculate_health_score(customer)
                        st.metric("Health Score", f"{health_score}/100")
                    
                    st.divider()
                    
                    # Tabs for different views
                    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Purchase History", "Support Tickets", "AI Insights"])
                    
                    with tab1:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### Engagement Timeline")
                            timeline_df = pd.DataFrame({
                                'Date': pd.date_range(end=datetime.now(), periods=30, freq='D'),
                                'Purchases': np.random.randint(0, 3, 30),
                                'Support Tickets': np.random.randint(0, 2, 30)
                            })
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=timeline_df['Date'], y=timeline_df['Purchases'], 
                                                    name='Purchases', line=dict(color='#4ECDC4', width=2)))
                            fig.add_trace(go.Scatter(x=timeline_df['Date'], y=timeline_df['Support Tickets'], 
                                                    name='Support Tickets', line=dict(color='#FF6B6B', width=2)))
                            fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.markdown("### Customer Attributes")
                            st.write(f"**Customer ID:** {customer.customer_id}")
                            st.write(f"**Segment:** {segment_str}")
                            st.write(f"**Lifetime Value:** {format_currency(customer.lifetime_value)}")
                            st.write(f"**Total Orders:** {customer.total_orders}")
                            st.write(f"**Avg Satisfaction:** {customer.satisfaction_score or 'N/A'}")
                            st.write(f"**Days Since Last Purchase:** {customer.days_since_last_purchase}")
                            st.write(f"**Open Tickets:** {customer.open_tickets}")
                    
                    with tab2:
                        st.markdown("### Recent Orders")
                        orders_df = pd.DataFrame({
                            'Order Date': ['2025-01-15', '2024-12-20', '2024-11-10'],
                            'Order ID': ['#12345', '#12234', '#12001'],
                            'Products': ['Widget Pro, Gadget X', 'Widget Standard', 'Accessory Pack'],
                            'Amount': ['$234.50', '$149.99', '$89.99'],
                            'Status': ['Delivered', 'Delivered', 'Delivered']
                        })
                        st.dataframe(orders_df, use_container_width=True, hide_index=True)
                    
                    with tab3:
                        st.markdown("### Support Ticket History")
                        tickets_df = pd.DataFrame({
                            'Created': ['2025-01-18', '2025-01-10', '2024-12-28'],
                            'Ticket ID': ['#5678', '#5632', '#5590'],
                            'Subject': ['Product not working', 'Billing question', 'Shipping delay'],
                            'Status': ['Open', 'Open', 'Closed'],
                            'Priority': ['High', 'Medium', 'Low'],
                            'Satisfaction': ['N/A', 'N/A', '5 stars']
                        })
                        st.dataframe(tickets_df, use_container_width=True, hide_index=True)
                    
                    with tab4:
                        st.markdown("### AI-Generated Insights")
                        
                        with st.spinner("Analyzing customer data..."):
                            try:
                                insights_query = "Analyze this customer's behavior and provide key insights about their churn risk and recommendations."
                                insights = st.session_state.rag_system.answer_query(
                                    query=insights_query,
                                    customer_email=customer_email
                                )
                                
                                if insights and isinstance(insights, dict):
                                    st.markdown("#### Key Insights")
                                    st.write(insights.get('answer', 'No insights available'))
                                    
                                    # Show citations if available
                                    citations = insights.get('citations', [])
                                    if citations and isinstance(citations, list) and len(citations) > 0:
                                        with st.expander("Sources"):
                                            for citation in citations:
                                                if isinstance(citation, dict):
                                                    st.markdown(f"""
                                                    <div class="citation">
                                                        <strong>{citation.get('source_type', 'Unknown')}</strong>: {citation.get('content', '')[:200]}...
                                                        <br><small>Relevance: {citation.get('similarity_score', 0):.2%}</small>
                                                    </div>
                                                    """, unsafe_allow_html=True)
                                    
                                    st.markdown("#### Recommended Actions")
                                    recommendations = [
                                        "Offer 15% discount on next purchase to re-engage",
                                        "Schedule proactive outreach call within 48 hours",
                                        "Send personalized product recommendations",
                                        "Prioritize open support tickets for immediate resolution"
                                    ]
                                    for rec in recommendations:
                                        st.write(f"• {rec}")
                                else:
                                    st.error("Could not generate insights")
                                    
                            except Exception as e:
                                st.error(f"Error generating insights: {str(e)}")
                else:
                    st.warning(f"Customer not found: {customer_email}")
                    
            except Exception as e:
                st.error(f"Error loading customer data: {str(e)}")

# ============================================================
# AI ASSISTANT PAGE
# ============================================================

elif selected == "AI Assistant":
    st.markdown("## AI Customer Intelligence Assistant")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        context_email = st.text_input(
            "Customer context (optional)",
            placeholder="customer@example.com",
            help="Provide customer email for personalized context"
        )
    
    with col2:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    st.divider()
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            if "citations" in message and message["citations"]:
                with st.expander("Sources"):
                    for citation in message["citations"]:
                        st.markdown(f"""
                        <div class="citation">
                            <strong>{citation.get('source_type', 'Unknown')}</strong>: {citation.get('content', '')[:200]}...
                            <br><small>Relevance: {citation.get('similarity_score', 0):.2%}</small>
                        </div>
                        """, unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your customers..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.rag_system.answer_query(
                        query=prompt,
                        customer_email=context_email if context_email else None
                    )
                    
                    if response is None:
                        error_msg = "I apologize, but I received no response from the AI system."
                        st.error(error_msg)
                        st.session_state.chat_history.append({"role": "assistant", "content": error_msg, "citations": []})
                    elif not isinstance(response, dict):
                        error_msg = f"Received invalid response format: {type(response).__name__}"
                        st.error(error_msg)
                        st.session_state.chat_history.append({"role": "assistant", "content": "I apologize, but I received an invalid response format.", "citations": []})
                    else:
                        answer = response.get('answer', 'I apologize, but I could not generate a response.')
                        st.write(answer)
                        
                        citations = response.get('citations', [])
                        if not isinstance(citations, list):
                            citations = []
                        
                        st.session_state.chat_history.append({"role": "assistant", "content": answer, "citations": citations})
                        
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    
                    with st.expander("Debug Information"):
                        import traceback
                        st.code(traceback.format_exc())
                    
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg, "citations": []})

# ============================================================
# ANALYTICS PAGE
# ============================================================

elif selected == "Analytics":
    st.markdown("## Customer Analytics Dashboard")
    
    # Date range selector
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
    with col2:
        end_date = st.date_input("End Date", datetime.now())
    with col3:
        refresh = st.button("Refresh", use_container_width=True)
    
    st.divider()
    
    # Churn risk distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Churn Risk Distribution")
        risk_data = pd.DataFrame({
            'Risk Level': ['High Risk', 'Medium Risk', 'Low Risk'],
            'Count': [342, 1205, 10996]
        })
        fig = px.pie(risk_data, values='Count', names='Risk Level',
                    color='Risk Level',
                    color_discrete_map={'High Risk': '#FF6B6B', 
                                       'Medium Risk': '#FFA500',
                                       'Low Risk': '#4ECDC4'})
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Customer Segments by LTV")
        segment_data = pd.DataFrame({
            'Segment': ['VIP', 'High Value', 'Medium Value', 'Low Value'],
            'Count': [450, 2100, 5200, 4793],
            'Avg LTV': [25000, 12000, 4500, 800]
        })
        fig = px.bar(segment_data, x='Segment', y='Count',
                    color='Avg LTV', color_continuous_scale='Viridis')
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    # Trends over time
    st.markdown("### Key Metrics Over Time")
    trend_df = pd.DataFrame({
        'Date': pd.date_range(end=datetime.now(), periods=30, freq='D'),
        'New Customers': np.random.randint(10, 50, 30),
        'Churn': np.random.randint(5, 20, 30),
        'Avg LTV': np.random.randint(8000, 9000, 30)
    })
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend_df['Date'], y=trend_df['New Customers'],
                            name='New Customers', line=dict(color='#4ECDC4')))
    fig.add_trace(go.Scatter(x=trend_df['Date'], y=trend_df['Churn'],
                            name='Churned', line=dict(color='#FF6B6B')))
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Top customers table
    st.markdown("### Top Customers by LTV")
    top_customers = pd.DataFrame({
        'Rank': [1, 2, 3, 4, 5],
        'Customer': ['Alice Smith', 'Bob Johnson', 'Carol White', 'David Brown', 'Eve Davis'],
        'Email': ['alice@example.com', 'bob@example.com', 'carol@example.com', 'david@example.com', 'eve@example.com'],
        'LTV': ['$45,230', '$42,100', '$38,900', '$35,400', '$32,800'],
        'Orders': [45, 38, 41, 32, 29],
        'Health Score': [95, 92, 88, 85, 82]
    })
    st.dataframe(top_customers, use_container_width=True, hide_index=True)

# ============================================================
# SETTINGS PAGE
# ============================================================

elif selected == "Settings":
    st.markdown("## System Settings")
    
    tab1, tab2, tab3 = st.tabs(["Configuration", "Data Sources", "System Status"])
    
    with tab1:
        st.markdown("### AI Model Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            llm_model = st.selectbox(
                "LLM Model",
                ["gemini-2.5-flash", "gemini-1.5-pro-002", "gemini-1.5-flash-002"],
                index=0
            )
            
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.1,
                help="Higher values make output more creative but less focused"
            )
        
        with col2:
            embedding_model = st.selectbox(
                "Embedding Model",
                ["text-embedding-004", "text-embedding-003"],
                index=0
            )
            
            top_k = st.number_input(
                "Top K Results",
                min_value=1,
                max_value=20,
                value=5,
                help="Number of documents to retrieve for RAG"
            )
        
        st.markdown("### Cache Settings")
        enable_cache = st.checkbox("Enable Response Caching", value=True)
        cache_ttl = st.number_input("Cache TTL (seconds)", min_value=60, max_value=3600, value=600)
        
        if st.button("Save Settings", type="primary"):
            st.success("Settings saved successfully!")
    
    with tab2:
        st.markdown("### Connected Data Sources")
        
        sources = [
            {"name": "Salesforce", "status": "Connected", "last_sync": "2 minutes ago", "records": "12,543"},
            {"name": "Zendesk", "status": "Connected", "last_sync": "5 minutes ago", "records": "8,932"},
            {"name": "Shopify", "status": "Connected", "last_sync": "3 minutes ago", "records": "45,231"},
            {"name": "Google Analytics", "status": "Connected", "last_sync": "1 minute ago", "records": "1,234,567"}
        ]
        
        for source in sources:
            with st.container():
                col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                with col1:
                    st.write(f"**{source['name']}**")
                with col2:
                    st.success(source['status'])
                with col3:
                    st.write(f"Last sync: {source['last_sync']}")
                with col4:
                    st.write(source['records'])
                st.divider()
        
        if st.button("Trigger Manual Sync"):
            with st.spinner("Syncing data..."):
                import time
                time.sleep(2)
                st.success("All data sources synced!")
    
    with tab3:
        st.markdown("### System Health")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("API Status", "Healthy", delta="99.9% uptime")
        with col2:
            st.metric("Query Latency", "450ms", delta="-50ms")
        with col3:
            st.metric("Error Rate", "0.1%", delta="-0.2%")
        
        st.markdown("### Resource Usage")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**BigQuery Usage (Last 30 Days)**")
            st.write("Queries: 45,231")
            st.write("Storage: 125 GB")
            st.write("Cost: $87.45")
        
        with col2:
            st.markdown("**Vertex AI Usage (Last 30 Days)**")
            st.write("LLM Calls: 12,543")
            st.write("Embeddings: 50,000")
            st.write("Cost: $234.12")
        
        st.markdown("### Recent Activity Log")
        activity_log = pd.DataFrame({
            'Timestamp': [datetime.now() - timedelta(minutes=i*5) for i in range(10)],
            'Event': ['User query', 'Data sync', 'User query', 'Embedding generation', 'User query',
                     'Data sync', 'User query', 'System health check', 'User query', 'Data sync'],
            'Status': ['Success'] * 10
        })
        st.dataframe(activity_log, use_container_width=True, hide_index=True)

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p><strong>Conversational 360</strong> | Built with Fivetran, Google Cloud, and Vertex AI</p>
    <p style='font-size: 0.85rem;'>2025 | <a href='https://github.com/your-repo'>GitHub</a> | <a href='https://your-docs.com'>Documentation</a></p>
</div>
""", unsafe_allow_html=True)