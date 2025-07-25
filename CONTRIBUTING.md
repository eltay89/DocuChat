# Contributing to DocuChat

We welcome contributions to DocuChat! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please treat all contributors with respect and create a welcoming environment for everyone.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Docker (optional, for containerized development)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/DocuChat.git
   cd DocuChat
   ```

## Development Setup

### Local Development

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

4. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

### Docker Development

1. Build the development image:
   ```bash
   docker-compose -f docker-compose.dev.yml build
   ```

2. Start the development environment:
   ```bash
   docker-compose -f docker-compose.dev.yml up
   ```

## Contributing Guidelines

### Types of Contributions

We welcome several types of contributions:

- **Bug Reports**: Help us identify and fix issues
- **Feature Requests**: Suggest new features or improvements
- **Code Contributions**: Implement new features or fix bugs
- **Documentation**: Improve or add documentation
- **Testing**: Add or improve tests

### Reporting Issues

When reporting issues, please include:

- Clear description of the problem
- Steps to reproduce the issue
- Expected vs. actual behavior
- Environment details (OS, Python version, etc.)
- Relevant logs or error messages

### Suggesting Features

For feature requests, please provide:

- Clear description of the proposed feature
- Use case and motivation
- Possible implementation approach
- Any relevant examples or references

## Pull Request Process

### Before Submitting

1. **Check existing issues**: Look for related issues or discussions
2. **Create an issue**: For significant changes, create an issue first
3. **Fork and branch**: Create a feature branch from `main`
4. **Follow conventions**: Use our coding standards and commit message format

### Submission Steps

1. **Create a branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes**: Implement your feature or fix

3. **Test thoroughly**:
   ```bash
   pytest tests/
   python -m docuchat.cli.main  # Manual testing
   ```

4. **Update documentation**: Add or update relevant documentation

5. **Commit changes**:
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create Pull Request**: Submit a PR with a clear description

### Pull Request Requirements

- [ ] Code follows our style guidelines
- [ ] Tests pass and coverage is maintained
- [ ] Documentation is updated
- [ ] Commit messages follow conventional format
- [ ] No merge conflicts with main branch
- [ ] PR description clearly explains changes

## Coding Standards

### Python Style

- Follow PEP 8 style guidelines
- Use type hints for all functions and methods
- Maximum line length: 88 characters (Black formatter)
- Use descriptive variable and function names

### Code Quality Tools

We use several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **pytest**: Testing

### Pre-commit Hooks

Pre-commit hooks automatically run quality checks:

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### Commit Message Format

Use conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(core): add hybrid search functionality
fix(cli): resolve file monitoring issue
docs(readme): update installation instructions
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=docuchat

# Run specific test file
pytest tests/test_document_processor.py

# Run with verbose output
pytest -v
```

### Writing Tests

- Write tests for all new functionality
- Maintain or improve test coverage
- Use descriptive test names
- Follow AAA pattern (Arrange, Act, Assert)
- Use fixtures for common test data

### Test Structure

```python
def test_feature_description():
    # Arrange
    input_data = create_test_data()
    processor = DocumentProcessor()
    
    # Act
    result = processor.process(input_data)
    
    # Assert
    assert result.status == "success"
    assert len(result.documents) > 0
```

## Documentation

### Types of Documentation

- **API Documentation**: Docstrings for all public functions/classes
- **User Guides**: How-to guides and tutorials
- **Architecture**: System design and component interaction
- **Contributing**: This document and development guides

### Documentation Standards

- Use Google-style docstrings
- Include examples in docstrings
- Keep documentation up-to-date with code changes
- Use clear, concise language

### Building Documentation

```bash
# Install documentation dependencies
pip install -r docs/requirements.txt

# Build documentation
cd docs
make html

# Serve locally
python -m http.server 8000 -d _build/html
```

## Adding New Features

### Document Processors

To add support for a new document format:

1. Create a new processor class inheriting from `BaseDocumentProcessor`
2. Implement required methods (`can_process`, `process`)
3. Register the processor in the factory
4. Add tests and documentation

### Tools

To add a new tool:

1. Create a new tool class inheriting from `BaseTool`
2. Implement required methods (`name`, `description`, `execute`)
3. Place in the `tools/` directory
4. Add tests and documentation

### Vector Stores

To add a new vector store backend:

1. Implement the vector store interface
2. Add configuration options
3. Register in the factory
4. Add tests and documentation

## Release Process

### Version Numbering

We follow Semantic Versioning (SemVer):

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Steps

1. Update version numbers
2. Update CHANGELOG.md
3. Create release branch
4. Run full test suite
5. Create GitHub release
6. Deploy to package repositories

## Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Email**: For security-related issues

### Resources

- [Project Documentation](./README.md)
- [Architecture Guide](./ARCHITECTURE.md)
- [API Documentation](./API.md)
- [Installation Guide](./INSTALLATION.md)

## Recognition

We appreciate all contributions and will:

- Add contributors to the README
- Mention significant contributions in release notes
- Provide feedback and support for contributors

Thank you for contributing to DocuChat!
