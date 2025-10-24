#!/bin/bash
# Fivetran Setup Script for Customer 360 Hackathon
# Execute this to set up all connectors in 30 minutes

# Prerequisites:
# - Fivetran account with API access
# - Google Cloud project with BigQuery enabled
# - API credentials for source systems

# Set environment variables
export FIVETRAN_API_KEY="your_api_key"
export FIVETRAN_API_SECRET="your_api_secret"
export GCP_PROJECT_ID="your-gcp-project"
export BQ_DATASET="customer_360"

# 1. Create BigQuery destination in Fivetran
echo "Setting up BigQuery destination..."
curl -X POST "https://api.fivetran.com/v1/destinations" \
  -u "$FIVETRAN_API_KEY:$FIVETRAN_API_SECRET" \
  -H "Content-Type: application/json" \
  -d '{
    "service": "google_cloud_bigquery",
    "region": "US",
    "config": {
      "project_id": "'$GCP_PROJECT_ID'",
      "dataset_id": "'$BQ_DATASET'",
      "service_account_key": "<your_service_account_json>"
    }
  }'

# 2. Set up Salesforce connector
echo "Setting up Salesforce connector..."
curl -X POST "https://api.fivetran.com/v1/groups/{group_id}/connectors" \
  -u "$FIVETRAN_API_KEY:$FIVETRAN_API_SECRET" \
  -H "Content-Type: application/json" \
  -d '{
    "service": "salesforce",
    "config": {
      "domain": "login.salesforce.com",
      "client_id": "your_sf_client_id",
      "client_secret": "your_sf_client_secret",
      "security_token": "your_sf_token"
    },
    "sync_frequency": 15,
    "schema_config": {
      "schemas": {
        "salesforce": {
          "tables": {
            "Account": {"enabled": true},
            "Contact": {"enabled": true},
            "Opportunity": {"enabled": true},
            "Case": {"enabled": true},
            "User": {"enabled": true}
          }
        }
      }
    }
  }'

# 3. Set up Zendesk connector
echo "Setting up Zendesk connector..."
curl -X POST "https://api.fivetran.com/v1/groups/{group_id}/connectors" \
  -u "$FIVETRAN_API_KEY:$FIVETRAN_API_SECRET" \
  -H "Content-Type: application/json" \
  -d '{
    "service": "zendesk",
    "config": {
      "subdomain": "your-company",
      "api_token": "your_zendesk_api_token",
      "email": "admin@company.com"
    },
    "sync_frequency": 15,
    "schema_config": {
      "schemas": {
        "zendesk": {
          "tables": {
            "tickets": {"enabled": true},
            "ticket_comments": {"enabled": true},
            "users": {"enabled": true},
            "organizations": {"enabled": true},
            "satisfaction_ratings": {"enabled": true}
          }
        }
      }
    }
  }'

# 4. Set up Shopify connector (or any e-commerce platform)
echo "Setting up Shopify connector..."
curl -X POST "https://api.fivetran.com/v1/groups/{group_id}/connectors" \
  -u "$FIVETRAN_API_KEY:$FIVETRAN_API_SECRET" \
  -H "Content-Type: application/json" \
  -d '{
    "service": "shopify",
    "config": {
      "shop": "your-shop.myshopify.com",
      "api_key": "your_shopify_api_key",
      "api_secret": "your_shopify_api_secret"
    },
    "sync_frequency": 15,
    "schema_config": {
      "schemas": {
        "shopify": {
          "tables": {
            "customer": {"enabled": true},
            "order": {"enabled": true},
            "product": {"enabled": true},
            "abandoned_checkouts": {"enabled": true}
          }
        }
      }
    }
  }'

# 5. Set up Google Analytics connector
echo "Setting up Google Analytics connector..."
curl -X POST "https://api.fivetran.com/v1/groups/{group_id}/connectors" \
  -u "$FIVETRAN_API_KEY:$FIVETRAN_API_SECRET" \
  -H "Content-Type: application/json" \
  -d '{
    "service": "google_analytics_4",
    "config": {
      "property_id": "your_ga4_property_id",
      "client_email": "your_service_account@project.iam.gserviceaccount.com",
      "private_key": "your_private_key"
    },
    "sync_frequency": 60
  }'

# 6. Trigger initial sync for all connectors
echo "Triggering initial sync..."
# You can trigger syncs via API or Fivetran dashboard

echo " All Fivetran connectors configured!"
echo "Data will start flowing to BigQuery dataset: $BQ_DATASET"
echo "Initial sync typically takes 30-60 minutes"