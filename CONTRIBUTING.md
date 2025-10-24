# Contributing to Conversational 360

Thank you for your interest in contributing to Conversational 360! We welcome contributions from the community.

##  How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Screenshots (if applicable)
- Environment details (OS, Python version, etc.)

### Suggesting Features

We love feature suggestions! Please create an issue with:
- Clear description of the feature
- Use case and expected benefits
- Any implementation ideas you have

### Code Contributions

1. **Fork the repository**
   ```bash
   git clone https://github.com/your-username/conversational-360.git
   cd conversational-360
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```

3. **Set up development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. **Make your changes**
   - Write clean, readable code
   - Follow existing code style
   - Add tests for new features
   - Update documentation as needed

5. **Run tests**
   ```bash
   # Run all tests
   pytest tests/
   
   # Run with coverage
   pytest --cov=src tests/
   
   # Run linting
   flake8 src/
   black src/ --check
   mypy src/
   ```

6. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add amazing feature"
   ```
   
   We follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation changes
   - `style:` Code style changes (formatting)
   - `refactor:` Code refactoring
   - `test:` Adding or updating tests
   - `chore:` Maintenance tasks

7. **Push to your fork**
   ```bash
   git push origin feature/amazing-feature
   ```

8. **Open a Pull Request**
   - Go to the original repository
   - Click "New Pull Request"
   - Select your fork and branch
   - Fill out the PR template
   - Request review

##  Pull Request Guidelines

### Before Submitting

- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Commit messages follow conventional commits
- [ ] No merge conflicts with main branch

### PR Description Should Include

- Summary of changes
- Related issue numbers (if applicable)
- Screenshots/GIFs for UI changes
- Breaking changes (if any)
- Migration guide (if needed)

##  Testing Guidelines

### Writing Tests

```python
import pytest
from src.rag_system import Customer360RAGSystem

def test_customer_lookup():
    """Test customer lookup functionality"""
    rag_system = Customer360RAGSystem(
        project_id="test-project",
        dataset_id="test_dataset"
    )
    
    customer = rag_system.get_customer_360("test@example.com")
    
    assert customer is not None
    assert customer.email == "test@example.com"
```

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_rag_system.py

# Specific test
pytest tests/test_rag_system.py::test_customer_lookup

# With verbose output
pytest -v

# With coverage
pytest --cov=src --cov-report=html
```

##  Code Style

We use:
- **Black** for code formatting
- **Flake8** for linting
- **MyPy** for type checking
- **isort** for import sorting

### Format code before committing:

```bash
# Format with black
black src/ tests/

# Sort imports
isort src/ tests/

# Check with flake8
flake8 src/ tests/

# Type check with mypy
mypy src/
```

### Pre-commit Hooks

Install pre-commit hooks to automatically check code:

```bash
pip install pre-commit
pre-commit install
```

##  Documentation

- Update docstrings for new functions/classes
- Follow Google-style docstrings
- Update README.md if adding new features
- Add examples for new functionality

### Docstring Example

```python
def calculate_health_score(customer: CustomerContext) -> int:
    """
    Calculate overall customer health score.
    
    Args:
        customer: Customer context with all relevant data
        
    Returns:
        Health score from 0-100, where higher is better
        
    Raises:
        ValueError: If customer data is invalid
        
    Examples:
        >>> customer = CustomerContext(...)
        >>> score = calculate_health_score(customer)
        >>> print(score)
        85
    """
    pass
```

##  Project Structure

```
src/
├── __init__.py
├── rag_system.py          # Core RAG logic
├── bigquery_client.py     # BigQuery operations
├── vertex_ai_client.py    # Vertex AI operations
├── data_models.py         # Pydantic models
└── utils.py               # Helper functions

tests/
├── test_rag_system.py
├── test_bigquery_client.py
└── test_utils.py

scripts/
├── setup_fivetran.sh
├── setup_bigquery_schema.py
└── generate_embeddings.py
```

##  Debugging

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Common Issues

**Issue: BigQuery authentication error**
```bash
# Set up application default credentials
gcloud auth application-default login
```

**Issue: Import errors**
```bash
# Ensure you're in the project root
export PYTHONPATH="${PYTHONPATH}:${PWD}"
```

##  Communication

- **Issues**: For bug reports and feature requests
- **Discussions**: For questions and general discussion
- **Pull Requests**: For code contributions

##  License

By contributing, you agree that your contributions will be licensed under the MIT License.

##  Thank You!

Every contribution, no matter how small, is valuable. Thank you for helping make Conversational 360 better!