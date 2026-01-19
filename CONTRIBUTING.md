# Contributing to RAG Q&A System

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Features](#suggesting-features)
- [Code Review Guidelines](#code-review-guidelines)

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive experience for everyone. We pledge to:

- Be respectful and considerate in all interactions
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

### Unacceptable Behavior

- Harassment, trolling, or discriminatory language
- Personal attacks or insults
- Publishing others' private information without permission
- Any conduct that would be inappropriate in a professional setting

## Getting Started

### Prerequisites

Before contributing, ensure you have:

- **Python 3.12+** installed
- **UV package manager** (recommended) or pip
- **Git** for version control
- **Docker** (optional, for testing containerization)
- API keys for:
  - OpenAI (for embeddings and LLM)
  - Qdrant Cloud (for vector storage)

### First-Time Contributors

If you're new to the project:

1. **Explore the codebase**: Read the [README.md](README.md) and familiarize yourself with the architecture
2. **Look for "good first issue" labels**: These are beginner-friendly tasks
3. **Ask questions**: Don't hesitate to ask in issues or discussions
4. **Start small**: Begin with documentation improvements or bug fixes

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/rag-qa-project.git
cd rag-qa-project

# Add upstream remote
git remote add upstream https://github.com/sourangshupal/rag-qa-project.git
```

### 2. Set Up Environment

**Using UV (Recommended):**
```bash
# Install UV if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync --extra dev

# Create .env file
cp .env.example .env  # Edit with your API keys
```

**Using pip:**
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Create .env file
cp .env.example .env  # Edit with your API keys
```

### 3. Verify Setup

```bash
# Run tests
uv run pytest  # or: pytest

# Check linting
uv run ruff check app/ tests/

# Check formatting
uv run black --check app/ tests/

# Run the app
uv run uvicorn app.main:app --reload
```

### 4. Create a Branch

```bash
# Always create a new branch for your work
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

## Coding Standards

### Python Style Guide

We follow **PEP 8** with some modifications:

- **Line length**: 100 characters (configured in Ruff)
- **Imports**: Organized using Ruff (isort-compatible)
- **Formatting**: Enforced by Black
- **Linting**: Enforced by Ruff

### Code Organization

```python
# Standard library imports
import os
from pathlib import Path

# Third-party imports
from fastapi import APIRouter, HTTPException
from langchain_openai import ChatOpenAI

# Local imports
from app.config import get_settings
from app.utils.logger import get_logger
```

### Naming Conventions

- **Functions/methods**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private methods**: `_leading_underscore`
- **Type hints**: Always use type hints for function signatures

### Documentation

#### Docstrings

Use Google-style docstrings for all public functions and classes:

```python
def query_with_sources(question: str, k: int = 4) -> dict[str, Any]:
    """Query the RAG system and return answer with source documents.

    Args:
        question: The user's question to answer
        k: Number of source documents to retrieve (default: 4)

    Returns:
        Dictionary containing:
            - answer: The generated answer
            - sources: List of source documents with metadata

    Raises:
        ValueError: If question is empty or k is invalid
        HTTPException: If LLM or vector store fails
    """
    pass
```

#### Comments

- Explain **why**, not **what** (code should be self-explanatory)
- Use comments for complex algorithms or business logic
- Avoid obvious comments

```python
# Good
# Use exponential backoff to handle rate limiting from OpenAI API
retry_count = 3

# Bad
# Set retry count to 3
retry_count = 3
```

### Async/Await

- Use `async`/`await` for I/O-bound operations (API calls, database queries)
- Prefer `async` methods for FastAPI endpoints
- Use `asyncio.to_thread()` for blocking operations in async context

### Error Handling

```python
# Good - Specific exception handling
try:
    result = await llm.ainvoke(prompt)
except OpenAIError as e:
    logger.error(f"OpenAI API error: {e}")
    raise HTTPException(status_code=503, detail="LLM service unavailable")
except Exception as e:
    logger.exception("Unexpected error during query")
    raise HTTPException(status_code=500, detail="Internal server error")

# Bad - Bare except
try:
    result = await llm.ainvoke(prompt)
except:
    pass
```

### Settings Management

**Important**: Never call `get_settings()` at module level. Always call it inside functions/methods:

```python
# Good
def __init__(self):
    settings = get_settings()
    self.model = settings.llm_model

# Bad - causes issues during testing and Docker builds
settings = get_settings()  # At module level

class MyClass:
    def __init__(self):
        self.model = settings.llm_model
```

## Testing

### Test Requirements

- **All new features** must include tests
- **Bug fixes** should include regression tests
- **Minimum coverage**: 60% (aim for higher)
- **Test types**: Unit tests, integration tests (no end-to-end yet)

### Writing Tests

#### Test Structure

```python
import pytest
from unittest.mock import MagicMock, AsyncMock

class TestQueryEndpoints:
    """Test query API endpoints."""

    def test_basic_query(self, client, mock_rag_chain):
        """Test basic query without sources."""
        response = client.post("/query", json={
            "question": "What is RAG?",
            "include_sources": False
        })

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert data["sources"] is None

    @pytest.mark.asyncio
    async def test_async_evaluation(self, mock_evaluator):
        """Test async evaluation with RAGAS."""
        result = await mock_evaluator.aevaluate(
            question="What is RAG?",
            answer="Retrieval-Augmented Generation",
            contexts=["RAG combines retrieval..."]
        )

        assert result["faithfulness"] > 0.8
        assert result["answer_relevancy"] > 0.8
```

#### Fixtures

Use fixtures for common test setup in `tests/conftest.py`:

```python
@pytest.fixture
def mock_rag_chain():
    """Mock RAG chain for testing."""
    with patch("app.api.routes.query.RAGChain") as mock:
        chain = MagicMock()

        # Use async functions for async methods
        async def mock_aquery(question):
            return "Test answer"

        chain.aquery = mock_aquery
        mock.return_value = chain
        yield chain
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_query.py

# Run with coverage
uv run pytest --cov=app --cov-report=html

# Run tests matching pattern
uv run pytest -k "test_query"

# Run with verbose output
uv run pytest -v

# Run only failed tests
uv run pytest --lf
```

### Test Coverage

Check coverage after running tests:

```bash
# View coverage in terminal
uv run pytest --cov=app --cov-report=term-missing

# Generate HTML report
uv run pytest --cov=app --cov-report=html
open htmlcov/index.html
```

## Pull Request Process

### Before Submitting

1. **Update from main**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run all checks**:
   ```bash
   # Linting
   uv run ruff check app/ tests/

   # Formatting
   uv run black app/ tests/

   # Type checking (optional)
   uv run mypy app/

   # Tests with coverage
   uv run pytest --cov=app --cov-fail-under=60
   ```

3. **Update documentation**:
   - Update README.md if adding new features
   - Add docstrings to new functions
   - Update CLAUDE.md if changing project structure

4. **Commit with clear messages**:
   ```bash
   git add -A
   git commit -m "Add feature: RAG streaming responses

   - Implement streaming endpoint in query.py
   - Add tests for streaming behavior
   - Update README with streaming examples"
   ```

### PR Title Format

Use conventional commits format:

- `feat: Add RAGAS evaluation metrics`
- `fix: Handle timeout errors in vector search`
- `docs: Update installation instructions`
- `test: Add integration tests for document upload`
- `refactor: Extract embedding logic to service`
- `chore: Update dependencies`

### PR Description Template

```markdown
## Description
Brief description of what this PR does.

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to change)
- [ ] Documentation update

## Changes Made
- List specific changes
- Include technical details
- Mention affected components

## Testing
- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Manual testing completed

## Screenshots (if applicable)
Add screenshots for UI changes or API responses.

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] Documentation updated
- [ ] No new warnings generated
- [ ] Tests added/updated and passing
```

### Review Process

1. **Automated checks** must pass:
   - Lint & Format (Ruff + Black)
   - Unit Tests (pytest)
   - Docker Build
   - Security Scan

2. **Code review** by maintainers:
   - Code quality and style
   - Test coverage
   - Documentation completeness
   - Performance considerations

3. **Feedback incorporation**:
   - Address all review comments
   - Push new commits (don't force push during review)
   - Request re-review when ready

4. **Merge**:
   - Maintainers will merge once approved
   - Squash and merge is preferred for feature PRs

## Reporting Bugs

### Before Reporting

1. **Search existing issues**: Your bug might already be reported
2. **Verify it's reproducible**: Can you consistently reproduce it?
3. **Check version**: Are you using the latest version?

### Bug Report Template

**Title**: Clear, concise description

**Description**:
- What you expected to happen
- What actually happened
- Impact/severity

**Steps to Reproduce**:
1. Go to '...'
2. Click on '...'
3. Scroll down to '...'
4. See error

**Environment**:
- OS: [e.g., macOS 14.0]
- Python: [e.g., 3.12.1]
- Dependencies: [run `uv pip list` or `pip freeze`]

**Error Messages/Logs**:
```
Paste relevant error messages or logs here
```

**Screenshots** (if applicable):
Attach screenshots showing the issue.

**Possible Solution** (optional):
If you have ideas on how to fix it.

## Suggesting Features

### Feature Request Template

**Title**: Clear, concise feature description

**Problem Statement**:
Describe the problem this feature would solve. Use user stories:
"As a [user type], I want [goal] so that [benefit]."

**Proposed Solution**:
Detailed description of how the feature would work.

**Alternatives Considered**:
Other approaches you've considered and why they don't work.

**Additional Context**:
- Use cases
- Examples from other projects
- Mockups/diagrams
- Performance considerations

**Willingness to Contribute**:
- [ ] I can implement this feature
- [ ] I can help test this feature
- [ ] I can only suggest the idea

## Code Review Guidelines

### For Reviewers

**Be constructive**:
- Focus on the code, not the person
- Explain **why** changes are needed
- Suggest specific improvements
- Praise good solutions

**Review checklist**:
- [ ] Code is readable and maintainable
- [ ] Follows project conventions
- [ ] Tests are comprehensive
- [ ] Documentation is updated
- [ ] No security vulnerabilities
- [ ] Performance is acceptable
- [ ] Error handling is appropriate

### For Contributors

**Receiving feedback**:
- Don't take criticism personally
- Ask for clarification if needed
- Learn from the feedback
- Thank reviewers for their time

**Responding to reviews**:
- Address all comments
- Explain your reasoning if disagreeing
- Mark resolved comments
- Request re-review when ready

## Git Workflow

### Branch Naming

- `feature/feature-name` - New features
- `fix/bug-description` - Bug fixes
- `docs/what-changed` - Documentation
- `refactor/what-changed` - Code refactoring
- `test/what-added` - Test additions

### Commit Messages

Follow conventional commits:

```bash
# Good
feat: add streaming response support
fix: handle empty context in RAGAS evaluation
docs: update API endpoint examples
test: add integration tests for document upload

# Bad
updated stuff
fix bug
changes
```

**Detailed commit message**:
```
feat: add LangSmith observability integration

- Add LangSmith tracing configuration
- Update environment variable handling
- Add documentation for observability setup

Implements automatic tracing for:
- LLM calls with token usage and costs
- RAG chain execution steps
- Vector store operations
- RAGAS evaluation metrics

Closes #123
```

### Keeping Your Fork Updated

```bash
# Fetch changes from upstream
git fetch upstream

# Update your main branch
git checkout main
git merge upstream/main
git push origin main

# Rebase your feature branch
git checkout feature/your-feature
git rebase main
```

## Additional Resources

- [README.md](README.md) - Project overview and setup
- [CLAUDE.md](CLAUDE.md) - Project instructions for AI assistants
- [AWS_DEPLOYMENT_GUIDE.md](docs/AWS_DEPLOYMENT_GUIDE.md) - Deployment instructions
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [LangChain Documentation](https://python.langchain.com/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)

## Questions?

- **Discussions**: Use GitHub Discussions for general questions
- **Issues**: Open an issue for bugs or feature requests
- **Email**: Contact maintainers for private concerns

---

Thank you for contributing to the RAG Q&A System! Your efforts help make this project better for everyone.
