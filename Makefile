# Conversational 360 - Makefile for common development tasks

.PHONY: help install setup run test lint format clean docker-build docker-run

# Default target
help:
	@echo "Conversational 360 - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install Python dependencies"
	@echo "  make setup           Complete project setup"
	@echo ""
	@echo "Development:"
	@echo "  make run             Run Streamlit app"
	@echo "  make test            Run tests"
	@echo "  make lint            Run linting checks"
	@echo "  make format          Format code"
	@echo ""
	@echo "Database:"
	@echo "  make create-schema   Create BigQuery schema"
	@echo "  make generate-embeddings  Generate embeddings"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build    Build Docker image"
	@echo "  make docker-run      Run Docker container"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean           Clean cache and temp files"

# Install dependencies
install:
	@echo " Installing dependencies..."
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo " Dependencies installed"

# Complete setup
setup: install
	@echo "Setting up project..."
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env file - please configure it"; \
	fi
	@mkdir -p logs cache data/examples
	@echo " Setup complete"

# Run Streamlit app
run:
	@echo " Starting Streamlit app..."
	streamlit run app.py

# Run tests
test:
	@echo " Running tests..."
	pytest tests/ -v --cov=src --cov-report=html
	@echo " Tests complete. Coverage report: htmlcov/index.html"

# Run linting
lint:
	@echo " Running linting checks..."
	flake8 src/ tests/ --max-line-length=100
	mypy src/
	@echo " Linting complete"

# Format code
format:
	@echo " Formatting code..."
	black src/ tests/ app.py
	isort src/ tests/ app.py
	@echo "Formatting complete"

# Create BigQuery schema
create-schema:
	@echo "🗄️  Creating BigQuery schema..."
	python scripts/setup_bigquery_schema.py
	@echo " Schema created"

# Generate embeddings
generate-embeddings:
	@echo " Generating embeddings..."
	python scripts/generate_embeddings.py --table all --batch-size 5
	@echo " Embeddings generated"

# Generate embeddings (with index creation)
generate-embeddings-with-index:
	@echo "Generating embeddings and creating indexes..."
	python scripts/generate_embeddings.py --table all --batch-size 5 --create-index
	@echo "Embeddings and indexes created"

# Show embedding statistics
embedding-stats:
	@echo "Fetching embedding statistics..."
	python scripts/generate_embeddings.py --stats-only

# Build Docker image
docker-build:
	@echo "Building Docker image..."
	docker build -t conversational-360:latest .
	@echo "Docker image built"

# Run Docker container
docker-run:
	@echo "Running Docker container..."
	docker-compose up -d
	@echo " Container started at http://localhost:8501"

# Stop Docker container
docker-stop:
	@echo " Stopping Docker container..."
	docker-compose down
	@echo " Container stopped"

# View Docker logs
docker-logs:
	docker-compose logs -f app

# Clean cache and temporary files
clean:
	@echo " Cleaning cache and temp files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete
	rm -rf htmlcov/ .coverage cache/ logs/*.log
	@echo " Cleanup complete"

# Run pre-commit hooks
pre-commit:
	@echo " Running pre-commit hooks..."
	pre-commit run --all-files

# Initialize pre-commit
init-pre-commit:
	@echo " Installing pre-commit hooks..."
	pip install pre-commit
	pre-commit install
	@echo " Pre-commit hooks installed"

# Run security check
security-check:
	@echo " Running security checks..."
	pip install safety bandit
	safety check
	bandit -r src/
	@echo " Security check complete"

# Generate documentation
docs:
	@echo " Generating documentation..."
	pip install mkdocs mkdocs-material
	mkdocs build
	@echo " Documentation generated in site/"

# Serve documentation
docs-serve:
	@echo " Serving documentation..."
	mkdocs serve

# Update dependencies
update-deps:
	@echo " Updating dependencies..."
	pip install --upgrade -r requirements.txt
	pip freeze > requirements.txt
	@echo " Dependencies updated"

# Check for outdated dependencies
check-deps:
	@echo " Checking for outdated dependencies..."
	pip list --outdated