# DocuChat API Documentation

## Overview
DocuChat provides both CLI and programmatic interfaces for document processing and chat functionality.

## Core Components

### Document Processing
- Multi-format support (PDF, DOCX, TXT, MD, HTML, PPTX, CSV, JSON)
- OCR capabilities for image-based documents
- Advanced text chunking and preprocessing

### Vector Store
- ChromaDB integration
- Hybrid search (dense + sparse)
- Cross-encoder reranking
- Metadata management

### Chat Interface
- Streaming responses
- Context-aware conversations
- Tool integration
- Real-time file monitoring

## Usage Examples

### CLI Interface
```bash
python -m docuchat.cli.main
```

### Programmatic Usage
```python
from docuchat.core import DocumentProcessor, VectorStore
from docuchat.chat import ChatEngine

# Initialize components
processor = DocumentProcessor()
vector_store = VectorStore()
chat_engine = ChatEngine(vector_store)

# Process documents
documents = processor.process_directory("./documents")
vector_store.add_documents(documents)

# Chat
response = chat_engine.chat("What is the main topic?")
print(response)
```

## Configuration

See `config/config.yaml` for detailed configuration options.

## Error Handling

All components implement comprehensive error handling with detailed logging.
