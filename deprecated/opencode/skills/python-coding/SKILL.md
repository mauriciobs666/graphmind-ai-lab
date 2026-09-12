---
name: python-coding
description: Python coding assistant for writing, testing, and maintaining Python code following best practices. Use for creating Python files, debugging, pytest testing, type hints, and Python-specific refactoring.
license: Apache-2.0
compatibility: opencode
metadata:
  audience: developers
  workflow: coding
---

## What I do

- Write and edit Python code with proper syntax and idiomatic patterns
- Create and run pytest tests for Python functions and modules
- Add type hints and annotations following PEP 484
- Debug Python code and fix common errors
- Refactor Python code following PEP 8 and best practices
- Set up virtual environments and manage dependencies
- Use list comprehensions, generators, and Python idioms effectively
- Work with dataclasses, decorators, and context managers

## When to use me

Use this when:
- Writing new Python files or modules
- Creating pytest test files for Python code
- Debugging Python errors or exceptions
- Adding type hints to existing Python code
- Refactoring Python code for readability and performance
- Setting up Python projects with proper structure
- Converting code to use Python idioms (list comprehensions, etc.)

## Prerequisites

- Python 3.8+ installed
- virtualenv or venv available for environment management
- pytest installed for testing

## Python Conventions

### File Structure
```
project/
├── src/              # Source code
├── tests/            # Test files
├── pyproject.toml     # Project config (preferred)
└── README.md
```

### Code Style
- Follow PEP 8 for formatting
- Use type hints for function parameters and return values
- Docstrings in Google, NumPy, or Sphinx style
- Maximum line length: 88 characters (Black default)

### Testing
- Use pytest for all tests
- Test file naming: `test_<module>.py`
- Use fixtures for shared test setup
- Aim for high coverage on critical paths

## Best Practices

### Type Hints
```python
def process_items(items: list[str], count: int) -> dict[str, int]:
    return {item: count for item in items}
```

### Dataclasses
```python
from dataclasses import dataclass

@dataclass
class Config:
    name: str
    timeout: int = 30
    debug: bool = False
```

### Error Handling
```python
try:
    result = risky_operation()
except ValueError as e:
    logger.error(f"Invalid value: {e}")
    raise
```

## Common Issues

### Import Errors
- Use relative imports within packages: `from . import module`
- Add `__init__.py` for package directories
- Check PYTHONPATH includes source directory

### Testing Issues
- Run tests from project root: `pytest tests/`
- Use `conftest.py` for shared fixtures
- Mock external dependencies in unit tests

### Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```
