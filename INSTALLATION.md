# DocuChat Installation Guide

This guide provides detailed instructions for installing and setting up DocuChat v2.0.

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation Methods](#installation-methods)
- [Configuration](#configuration)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements

- **Operating System**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **Memory**: 4GB RAM (8GB recommended)
- **Storage**: 2GB free space (more for document storage)
- **Internet**: Required for initial setup and model downloads

### Recommended Requirements

- **Python**: 3.9 or 3.10
- **Memory**: 8GB RAM or more
- **Storage**: 10GB free space
- **GPU**: CUDA-compatible GPU (optional, for enhanced performance)

## Installation Methods

### Method 1: Docker Installation (Recommended)

Docker provides the easiest and most reliable installation method.

#### Prerequisites

1. **Install Docker**:
   - **Windows**: [Docker Desktop for Windows](https://docs.docker.com/desktop/windows/install/)
   - **macOS**: [Docker Desktop for Mac](https://docs.docker.com/desktop/mac/install/)
   - **Linux**: [Docker Engine](https://docs.docker.com/engine/install/)

2. **Install Docker Compose** (usually included with Docker Desktop)

#### Quick Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/eltay89/DocuChat.git
   cd DocuChat
   ```

2. **Configure environment**:
   ```bash
   cp config/config.yaml.template config/config.yaml
   # Edit config.yaml with your settings
   ```

3. **Start DocuChat**:
   ```bash
   docker-compose up -d
   ```

4. **Access interfaces**:
   - CLI: `docker-compose exec cli python -m docuchat.cli.main`
   - Web: http://localhost:8000
   - Streamlit: http://localhost:8501

### Method 2: Local Python Installation

#### Step 1: Python Setup

1. **Verify Python version**:
   ```bash
   python --version
   # Should be 3.8 or higher
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv docuchat-env
   
   # Activate (Windows)
   docuchat-env\Scripts\activate
   
   # Activate (macOS/Linux)
   source docuchat-env/bin/activate
   ```

#### Step 2: Install DocuChat

1. **Clone repository**:
   ```bash
   git clone https://github.com/eltay89/DocuChat.git
   cd DocuChat
   ```

2. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Install DocuChat**:
   ```bash
   pip install -e .
   ```

#### Step 3: Additional Dependencies

For enhanced features, install optional dependencies:

```bash
# OCR support
pip install easyocr

# Advanced document processing
pip install unstructured[all-docs]

# Web interface
pip install fastapi uvicorn streamlit

# Development tools
pip install pytest black isort flake8 mypy
```

### Method 3: Package Installation (Coming Soon)

```bash
# PyPI installation (when available)
pip install docuchat

# Conda installation (when available)
conda install -c conda-forge docuchat
```

## Configuration

### Basic Configuration

1. **Copy configuration template**:
   ```bash
   cp config/config.yaml.template config/config.yaml
   ```

2. **Edit configuration**:
   ```yaml
   # config/config.yaml
   model:
     provider: "openrouter"  # or "openai", "anthropic"
     model_name: "anthropic/claude-3-haiku"
     api_key: "your-api-key-here"
   
   documents:
     input_directory: "./documents"
     supported_formats: ["pdf", "docx", "txt", "md"]
   
   embeddings:
     model_name: "sentence-transformers/all-MiniLM-L6-v2"
     dimension: 384
   
   rag:
     enabled: true
     chunk_size: 1000
     chunk_overlap: 200
   ```

### Environment Variables

Alternatively, use environment variables:

```bash
# API Configuration
export OPENROUTER_API_KEY="your-api-key"
export MODEL_PROVIDER="openrouter"
export MODEL_NAME="anthropic/claude-3-haiku"

# Document Processing
export DOCUMENTS_DIR="./documents"
export ENABLE_OCR="true"

# Vector Store
export VECTOR_STORE_PATH="./vector_store"
export EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
```

### Advanced Configuration

#### GPU Support

For GPU acceleration:

```yaml
embeddings:
  device: "cuda"  # or "cpu"
  batch_size: 32

processing:
  use_gpu: true
  gpu_memory_limit: "4GB"
```

#### Custom Models

```yaml
model:
  provider: "custom"
  base_url: "http://localhost:11434"  # Ollama example
  model_name: "llama2"
```

#### Enhanced Features

```yaml
enhanced:
  enabled: true
  ocr:
    enabled: true
    languages: ["en", "es", "fr"]
  
  reranking:
    enabled: true
    model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  
  hybrid_search:
    enabled: true
    dense_weight: 0.7
    sparse_weight: 0.3
```

## Verification

### Test Installation

1. **Basic functionality test**:
   ```bash
   python -c "import docuchat; print('DocuChat installed successfully!')"
   ```

2. **CLI test**:
   ```bash
   python -m docuchat.cli.main --help
   ```

3. **Component test**:
   ```bash
   python -c "
   from docuchat.core import DocumentProcessor
   processor = DocumentProcessor()
   print('Document processor initialized successfully!')
   "
   ```

### Sample Document Test

1. **Create test document**:
   ```bash
   mkdir -p documents
   echo "This is a test document for DocuChat." > documents/test.txt
   ```

2. **Run DocuChat**:
   ```bash
   python -m docuchat.cli.main
   ```

3. **Test query**:
   ```
   > What is in the test document?
   ```

### Docker Verification

```bash
# Check running containers
docker-compose ps

# View logs
docker-compose logs cli

# Test CLI access
docker-compose exec cli python -m docuchat.cli.main --version
```

## Troubleshooting

### Common Issues

#### Python Version Issues

**Problem**: "Python 3.8+ required"

**Solution**:
```bash
# Check Python version
python --version

# Install Python 3.8+ if needed
# Use pyenv for version management
curl https://pyenv.run | bash
pyenv install 3.10.0
pyenv global 3.10.0
```

#### Dependency Conflicts

**Problem**: Package installation failures

**Solution**:
```bash
# Clean installation
pip uninstall docuchat
rm -rf docuchat-env
python -m venv docuchat-env
source docuchat-env/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

#### Memory Issues

**Problem**: "Out of memory" errors

**Solution**:
```yaml
# Reduce batch sizes in config.yaml
embeddings:
  batch_size: 8  # Reduce from default 32

processing:
  chunk_size: 500  # Reduce from default 1000
  max_workers: 2   # Reduce parallel processing
```

#### Docker Issues

**Problem**: Docker containers not starting

**Solution**:
```bash
# Check Docker status
docker --version
docker-compose --version

# Rebuild containers
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# Check logs
docker-compose logs
```

#### API Key Issues

**Problem**: "Invalid API key" errors

**Solution**:
1. Verify API key in configuration
2. Check environment variables
3. Ensure proper provider selection
4. Test API key with curl:

```bash
# Test OpenRouter API
curl -X POST "https://openrouter.ai/api/v1/chat/completions" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "anthropic/claude-3-haiku", "messages": [{"role": "user", "content": "Hello"}]}'
```

#### File Permission Issues

**Problem**: Permission denied errors

**Solution**:
```bash
# Fix file permissions
chmod -R 755 documents/
chmod -R 755 vector_store/

# For Docker on Linux
sudo chown -R $USER:$USER documents/ vector_store/
```

### Performance Optimization

#### Slow Document Processing

1. **Enable GPU acceleration**:
   ```yaml
   embeddings:
     device: "cuda"
   ```

2. **Increase batch sizes**:
   ```yaml
   embeddings:
     batch_size: 64
   ```

3. **Use faster embedding models**:
   ```yaml
   embeddings:
     model_name: "sentence-transformers/all-MiniLM-L6-v2"  # Faster
     # vs "sentence-transformers/all-mpnet-base-v2"  # More accurate
   ```

#### Slow Search Performance

1. **Optimize vector store**:
   ```yaml
   vector_store:
     index_type: "hnsw"  # Faster than flat
     ef_construction: 200
     m: 16
   ```

2. **Enable hybrid search**:
   ```yaml
   enhanced:
     hybrid_search:
       enabled: true
   ```

### Getting Help

#### Support Channels

- **GitHub Issues**: [Report bugs and request features](https://github.com/eltay89/DocuChat/issues)
- **Discussions**: [Community support and questions](https://github.com/eltay89/DocuChat/discussions)
- **Documentation**: [Full documentation](./README.md)

#### Diagnostic Information

When reporting issues, include:

```bash
# System information
python --version
pip list | grep -E "(docuchat|torch|transformers|chromadb)"

# Docker information (if using Docker)
docker --version
docker-compose --version
docker-compose ps

# Configuration (remove sensitive data)
cat config/config.yaml

# Logs
tail -n 50 logs/docuchat.log
```

## Next Steps

After successful installation:

1. **Read the [User Guide](./README.md#usage)**
2. **Explore [Configuration Options](./README.md#configuration)**
3. **Check out [Examples](./README.md#examples)**
4. **Learn about [Advanced Features](./README.md#enhanced-features)**
5. **Join the [Community](https://github.com/eltay89/DocuChat/discussions)**

Welcome to DocuChat! 🚀
