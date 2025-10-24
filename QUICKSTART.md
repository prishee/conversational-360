# Quick Start Guide

Get Conversational 360 up and running in 30 minutes!

## Prerequisites

- Google Cloud Platform account ([free trial available](https://cloud.google.com/free))
- Fivetran account ([free trial available](https://fivetran.com/signup))
- Python 3.11 or higher
- Git

## Step-by-Step Setup

### **Step 1: Clone Repository** (2 minutes)

```bash
git clone https://github.com/your-username/conversational-360.git
cd conversational-360
```

### **Step 2: Set Up Google Cloud** (5 minutes)

```bash
# Install Google Cloud SDK
# https://cloud.google.com/sdk/docs/install

# Login to GCP
gcloud auth login

# Create new project (or use existing)
gcloud projects create conversational-360-demo
gcloud config set project conversational-360-demo

# Enable required APIs
gcloud services enable bigquery.googleapis.com
gcloud services enable aiplatform.googleapis.com
gcloud services enable storage.googleapis.com

# Set up authentication
gcloud auth application-default login
```

### **Step 3: Create BigQuery Dataset** (2 minutes)

```bash
# Create dataset
bq mk --location=US customer_360
```

### **Step 4: Configure Fivetran** (10 minutes)

1. **Sign up at [fivetran.com](https://fivetran.com)**

2. **Add BigQuery as destination:**
   - Dashboard → Destinations → Add Destination
   - Select "Google BigQuery"
   - Upload GCP service account JSON
   - Dataset: `customer_360`

3. **Add connectors** (choose based on your data):
   - Salesforce (CRM data)
   - Zendesk (Support tickets)
   - Shopify (E-commerce)
   - Google Analytics 4 (Web behavior)

4. **Trigger initial sync** and wait for completion (~20-30 minutes for historical data)

### **Step 5: Install Python Dependencies** (3 minutes)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **Step 6: Configure Environment** (2 minutes)

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your details
nano .env  # or use your preferred editor
```

**Required variables:**
```bash
GCP_PROJECT_ID=your-project-id
BQ_DATASET=customer_360
GCP_REGION=us-central1
```

### **Step 7: Create BigQuery Schema** (3 minutes)

```bash
# Run SQL setup scripts
python scripts/setup_bigquery_schema.py

# Or manually run SQL files
bq query --use_legacy_sql=false < sql/01_create_customer_360_view.sql
bq query --use_legacy_sql=false < sql/02_create_embeddings_tables.sql
```

### **Step 8: Generate Embeddings** (10-30 minutes)

```bash
# Start embedding generation
python scripts/generate_embeddings.py \
  --table all \
  --batch-size 5 \
  --create-index

# Monitor progress
# This will process all support tickets and products
```

**For testing** (faster - processes only 100 documents):
```bash
python scripts/generate_embeddings.py --table all --limit 100
```

### **Step 9: Launch Application** (1 minute)

```bash
# Start Streamlit app
streamlit run app.py

# Opens automatically at http://localhost:8501
```

---

## You're Done!

Your Conversational 360 platform is now running!

### **Try These First:**

1. **Customer Lookup:**
   - Click "Customer Lookup"
   - Enter a customer email from your data
   - View the complete 360° profile

2. **AI Assistant:**
   - Click "AI Assistant"
   - Ask: "Show me high-risk customers"
   - Or: "Why is customer X at risk?"

3. **Analytics:**
   - Click "Analytics"
   - View churn risk distribution
   - Explore customer segments

---

## Docker Alternative (Optional)

If you prefer Docker:

```bash
# Build image
docker build -t conversational-360:latest .

# Run container
docker run -p 8501:8501 \
  -v $(pwd)/.env:/app/.env \
  -v $(pwd)/credentials:/app/credentials \
  conversational-360:latest
```

Or use Docker Compose:

```bash
docker-compose up -d
```

---

## Troubleshooting

### **Issue: "Authentication Error"**

```bash
# Re-authenticate
gcloud auth application-default login

# Or set service account explicitly
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
```

### **Issue: "Table not found"**

```bash
# Verify dataset exists
bq ls

# Check if Fivetran sync completed
# Go to Fivetran dashboard and check sync status
```

### **Issue: "No embeddings found"**

```bash
# Check embedding generation status
python scripts/generate_embeddings.py --stats-only

# Regenerate if needed
python scripts/generate_embeddings.py --table all
```

### **Issue: "Import error"**

```bash
# Ensure you're in the project root directory
cd conversational-360

# Ensure virtual environment is activated
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Reinstall dependencies
pip install -r requirements.txt
```

---

## Next Steps

### **Customize Your Setup:**

1. **Add More Data Sources:**
   - Configure additional Fivetran connectors
   - Custom API integrations

2. **Fine-tune AI Models:**
   - Adjust temperature in settings
   - Modify top-K retrieval count
   - Customize prompts in `src/utils.py`

3. **Enhance UI:**
   - Modify `app.py` for custom branding
   - Add new dashboard views
   - Create custom reports

### **Production Deployment:**

See [docs/deployment.md](docs/deployment.md) for:
- Cloud Run deployment
- Kubernetes setup
- CI/CD pipelines
- Monitoring and alerting

---

## Need Help?

- **Documentation:** [Full Docs](docs/)
- **Issues:** [GitHub Issues](https://github.com/your-username/conversational-360/issues)
- **Community:** [Discussions](https://github.com/your-username/conversational-360/discussions)

---

## Estimated Setup Time

| Step | Time |
|------|------|
| Clone & GCP Setup | 10 min |
| Fivetran Configuration | 10 min |
| Python Setup | 5 min |
| Schema Creation | 3 min |
| **Initial Fivetran Sync** | **20-30 min** |
| Embedding Generation | 10-30 min |
| Launch App | 1 min |
| **Total** | **~60-90 min** |

*Most time is spent waiting for data sync and embedding generation to complete*

---

Happy building! 