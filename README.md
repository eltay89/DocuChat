# DocuChat 🤖📚

A powerful RAG (Retrieval-Augmented Generation) application that lets you chat with your documents using local language models. DocuChat processes your documents, creates vector embeddings, and provides intelligent responses based on your document content.

## Recent Updates 🆕

### Version 2.1 - Streaming Output & Enhanced User Experience
- **⚡ Real-time Streaming**: Character-by-character response generation for better user experience
- **🎛️ Flexible Streaming Control**: Configure streaming via config.yaml, CLI arguments, or use defaults
- **🔧 Enhanced Configuration**: Streaming enabled by default with easy override options
- **📊 Status Display**: Visual indicators for current streaming configuration
- **🚀 Improved Performance**: Optimized response generation with streaming capabilities

### Version 2.0 - Enhanced GGUF Auto-Detection
- **🎯 Robust GGUF Processing**: Improved metadata reading with fallback mechanisms
- **🔧 Enhanced Error Handling**: Better diagnostics for model loading issues
- **📊 Advanced Chat Templates**: Auto-detecting chat templates with comprehensive format support
- **🚀 Performance Optimizations**: Faster document processing and embedding generation
- **🧪 Comprehensive Testing**: New test suite for GGUF functionality and chat templates

## Features ✨

### Advanced AI Capabilities
- **⚡ Real-time Streaming**: Character-by-character response generation with configurable streaming
- **🎯 Auto-Detecting Chat Templates**: Automatically detects and applies the correct chat template format
- **🔧 Robust GGUF Processing**: Enhanced metadata reading with comprehensive error handling
- **🤖 Local LLM Support**: Uses GGUF format models via llama-cpp-python
- **📚 Multiple Document Formats**: Supports PDF, DOCX, TXT, and Markdown files
- **🔍 Vector Search**: Efficient document retrieval using ChromaDB and sentence transformers
- **💬 Interactive Chat**: Real-time conversation interface with streaming output
- **⚙️ Flexible Configuration**: Support for both YAML configuration files and command-line arguments
- **🎨 Customizable**: Multiple chat templates and configurable parameters
- **⚡ Performance Optimized**: Chunking, caching, and efficient embedding generation

## Installation 🚀

### Prerequisites
- Python 3.8 or higher
- At least 8GB RAM (16GB+ recommended for larger models)
- CUDA-compatible GPU (optional, for GPU acceleration)

### Install Dependencies
```bash
# Clone the repository
git clone https://github.com/eltay89/DocuChat.git
cd DocuChat

# Install required packages
pip install -r requirements.txt
```

### Download a Model
Download a GGUF format model from Hugging Face. Popular options:
```bash
# Example: Download a Llama 2 7B model
wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf
```
Or browse models at: https://huggingface.co/models?library=gguf

## Configuration 📋

DocuChat supports dual configuration through YAML files and command-line arguments, with command-line arguments taking precedence.

### YAML Configuration
The default configuration file is `config/config.yaml`. You can specify a custom config file using the `--config` argument.

#### Using Default Config
```bash
python docuchat.py
```

#### Using Custom Config
```bash
python docuchat.py --config my_config.yaml
```

#### Configuration Structure
The YAML configuration file supports the following sections:
- **model**: LLM settings (path, context length, sampling parameters)
- **documents**: Document processing settings (folder path, chunking)
- **embeddings**: Embedding model configuration
- **vector_store**: ChromaDB settings
- **rag**: Retrieval-augmented generation parameters
- **ui**: User interface and chat settings
- **logging**: Logging configuration
- **performance**: Performance optimization settings
- **advanced**: Advanced features including streaming configuration

#### Streaming Configuration
Streaming is configured in the `advanced.experimental` section of your YAML file:

```yaml
advanced:
  experimental:
    streaming: true  # Enable character-by-character output (default: true)
```

**Streaming Priority Order:**
1. **Command-line arguments** (`--streaming` or `--no-streaming`) - Highest priority
2. **YAML configuration** (`advanced.experimental.streaming`) - Medium priority  
3. **Default value** (`true`) - Lowest priority

**Streaming Benefits:**
- **Real-time feedback**: See responses as they're generated
- **Better user experience**: No waiting for complete response
- **Responsive interface**: Immediate visual feedback
- **Interruptible**: Can stop generation if needed

#### Mixed Configuration Example
You can combine YAML configuration with command-line overrides:
```bash
# Use config.yaml but override model and folder paths
python docuchat.py --model_path ./custom-model.gguf --folder_path ./my-docs
```

## Command Line Options 🛠️

### Configuration
- `--config`: Path to YAML configuration file (default: config/config.yaml)

### Model Options
- `--model_path`: Path to GGUF model file (overrides config file setting)
- `--n_ctx`: Context window size (default: 4096)
- `--temperature`: Sampling temperature (default: 0.7)
- `--max_tokens`: Maximum tokens to generate (default: 2048)

### Document Options
- `--folder_path`: Path to documents folder
- `--chunk_size`: Document chunk size (default: 1000)
- `--chunk_overlap`: Chunk overlap size (default: 200)

### RAG Options
- `--embedding_model`: Sentence transformer model (default: all-MiniLM-L6-v2)
- `--download_embedding_model`: Download embedding model from Hugging Face to embeddings/ folder
- `--n_retrieve`: Number of documents to retrieve (default: 5)
- `--no-rag`: Disable RAG (Retrieval-Augmented Generation) and use LLM only

### Streaming Options
- `--streaming`: Enable streaming output (character-by-character response generation)
- `--no-streaming`: Disable streaming output (show complete response at once)

### Other Options
- `--system_prompt`: Custom system prompt
- `--chat_template`: Chat template format (auto, chatml, llama2, alpaca)
- `--verbose`: Enable verbose logging
- `--query`: Single query mode (non-interactive)

## Usage Examples 📖

### Basic Usage with YAML Config
```bash
# Using default config file (config/config.yaml)
python docuchat.py

# Using custom config file
python docuchat.py --config my_config.yaml
```

### Command Line Only
```bash
# Specify all settings via command line
python docuchat.py --config "" --model_path ./models/llama-2-7b.gguf --folder_path ./documents
```

### Mixed Configuration
```bash
# Use YAML config but override specific settings
python docuchat.py --model_path ./custom-model.gguf --temperature 0.9 --verbose
```

### Chat Template Examples
```bash
# Use ChatML format
python docuchat.py --chat_template chatml

# Use Llama 2 format
python docuchat.py --chat_template llama2

# Use Alpaca format
python docuchat.py --chat_template alpaca
```

### Embedding Model Management
```bash
# Download a custom embedding model from Hugging Face
python docuchat.py --download_embedding_model "sentence-transformers/all-mpnet-base-v2"

# Use a downloaded custom embedding model
python docuchat.py --embedding_model "sentence-transformers--all-mpnet-base-v2"

# Use a different Hugging Face model directly
python docuchat.py --embedding_model "multi-qa-MiniLM-L6-cos-v1"
```

### Single Query Mode
```bash
# Process a single query without interactive mode
python docuchat.py --query "What is the main topic of the documents?"
```

### LLM-Only Mode (No RAG)
```bash
# Use the LLM without document retrieval
python docuchat.py --no-rag --model_path ./models/llama-2-7b.gguf

# Combine with single query for direct LLM responses
python docuchat.py --no-rag --query "Explain quantum computing" --model_path ./model.gguf
```

### Streaming Configuration Examples
```bash
# Enable streaming output (default behavior)
python docuchat.py --streaming

# Disable streaming for complete responses
python docuchat.py --no-streaming

# Use config file streaming setting
python docuchat.py  # Uses config.yaml advanced.experimental.streaming setting
```

### Advanced Usage
```bash
# High-performance setup with custom settings and streaming
python docuchat.py \
  --model_path ./models/llama-2-13b.gguf \
  --folder_path ./research_papers \
  --chunk_size 1500 \
  --chunk_overlap 300 \
  --n_retrieve 8 \
  --temperature 0.3 \
  --max_tokens 4096 \
  --streaming \
  --verbose
```

## Interactive Commands 💬

Once in interactive mode, you can use these commands:
- `help` - Show available commands and current configuration
- `quit`, `exit`, `bye` - Exit the application
- Any other text - Chat with your documents

### Status Information
The application displays current configuration including:
- **Model**: Currently loaded GGUF model
- **Documents**: Number of processed documents and chunks
- **Embedding Model**: Active embedding model
- **Streaming**: Current streaming status (Enabled/Disabled)
- **RAG Mode**: Whether document retrieval is active
- **Context Window**: Current context window size

## Embedding Models 🧠

### Default Model
DocuChat uses `all-MiniLM-L6-v2` as the default embedding model, which provides:
- Fast performance
- Good quality embeddings
- 384 dimensions
- Suitable for most document types

### Available Embedding Models
You can use various embedding models:

#### Hugging Face Models (Direct)
- `all-MiniLM-L6-v2` (default) - Fast, good quality, 384 dims
- `all-mpnet-base-v2` - Slower, better quality, 768 dims
- `multi-qa-MiniLM-L6-cos-v1` - Optimized for Q&A, 384 dims
- `paraphrase-multilingual-MiniLM-L12-v2` - Multilingual support, 384 dims

#### Local Models
Download models to the `embeddings/` folder for offline use:
```bash
# Download a model
python docuchat.py --download_embedding_model "sentence-transformers/all-mpnet-base-v2"

# Use the downloaded model
python docuchat.py --embedding_model "sentence-transformers--all-mpnet-base-v2"
```

#### Custom Paths
You can also specify absolute or relative paths:
```bash
python docuchat.py --embedding_model "./embeddings/my-custom-model"
python docuchat.py --embedding_model "/path/to/custom/model"
```

### Model Selection Guidelines
- **Speed Priority**: Use `all-MiniLM-L6-v2` (default)
- **Quality Priority**: Use `all-mpnet-base-v2`
- **Q&A Tasks**: Use `multi-qa-MiniLM-L6-cos-v1`
- **Multilingual**: Use `paraphrase-multilingual-MiniLM-L12-v2`
- **Offline Use**: Download models locally with `--download_embedding_model`

## Supported File Formats 📄
- **PDF** (.pdf) - Extracted using PyPDF2
- **Word Documents** (.docx, .doc) - Processed with python-docx
- **Text Files** (.txt) - Plain text
- **Markdown** (.md) - Markdown formatted text

## How It Works 🔧

### How It Works 🔧

### RAG Mode (Default)
1. **Document Processing**: Documents are loaded and split into overlapping chunks
2. **Embedding Generation**: Text chunks are converted to vector embeddings using sentence transformers
3. **Vector Storage**: Embeddings are stored in ChromaDB for efficient similarity search
4. **Query Processing**: User queries are embedded and matched against document chunks
5. **Context Retrieval**: Most relevant document chunks are retrieved based on similarity
6. **Response Generation**: Retrieved context is combined with the query and sent to the LLM
7. **Streaming Output**: Responses are generated token-by-token and displayed in real-time
8. **Chat Template Application**: Responses are formatted using the appropriate chat template

### LLM-Only Mode (--no-rag)
When using the `--no-rag` flag, DocuChat bypasses the document retrieval process and sends queries directly to the language model, functioning as a standard chatbot without document context.

### Streaming Implementation
DocuChat implements two response generation modes:

#### Streaming Mode (Default)
```python
def generate_response_streaming(self, prompt):
    for token in self.llm(prompt, stream=True):
        print(token['choices'][0]['text'], end='', flush=True)
```
- **Real-time Output**: Tokens are displayed as they're generated
- **Better UX**: Users see immediate feedback
- **Interruptible**: Generation can be stopped mid-stream

#### Non-Streaming Mode
```python
def generate_response(self, prompt):
    response = self.llm(prompt, stream=False)
    return response['choices'][0]['text']
```
- **Complete Response**: Full response generated before display
- **Batch Processing**: Useful for automated processing
- **Consistent Timing**: Predictable response delivery

### Vector Database Storage 💾
**Yes, this application stores a vector database!** The app uses ChromaDB to persistently store vector embeddings of your documents. Here's how:

- **Storage Location**: Vector database is stored in the `./chroma_db` directory (configurable in `config.yaml`)
- **Persistence**: The database persists between sessions, so you don't need to re-process documents every time
- **Benefits**: Faster startup times after initial processing, efficient similarity search, and reduced computational overhead
- **Management**: The database is automatically created and managed by ChromaDB

## Educational Guide: Understanding DocuChat 🎓

*This section is designed for junior students and developers learning about RAG systems, NLP, and AI applications.*

### Core Concepts Explained

#### What is RAG (Retrieval-Augmented Generation)?
RAG is a technique that combines:
- **Retrieval**: Finding relevant information from a knowledge base
- **Generation**: Using an AI model to create responses based on retrieved information
- **Streaming**: Real-time token-by-token response generation for better user experience

Think of it like an open-book exam where the AI can look up relevant information before answering your question, and then explains the answer as they think through it.

#### Why Use RAG with Streaming?
1. **Knowledge Limitation**: LLMs have a knowledge cutoff date and can't access real-time or private information
2. **Hallucination Reduction**: By grounding responses in actual documents, we reduce made-up information
3. **Domain Specificity**: Allows the AI to become an expert on your specific documents
4. **Cost Efficiency**: Cheaper than fine-tuning models on custom data
5. **Real-time Feedback**: Streaming provides immediate visual feedback during response generation
6. **Better UX**: Users don't wait for complete responses, improving perceived performance

### Key Libraries and Their Roles 📚

#### 1. **llama-cpp-python** - Local Language Model Engine
```python
from llama_cpp import Llama
```
**What it does**: Runs GGUF format language models locally on your machine
**Why we use it**: 
- Privacy (no data sent to external APIs)
- Cost-effective (no API fees)
- Offline capability
- Support for quantized models (smaller file sizes)

**How it works**: Converts GGUF model files into executable inference engines using optimized C++ code

#### 2. **ChromaDB** - Vector Database
```python
import chromadb
from chromadb.config import Settings
```
**What it does**: Stores and searches vector embeddings efficiently
**Why we use it**:
- Fast similarity search using cosine similarity
- Persistent storage (saves embeddings between sessions)
- Automatic indexing and optimization
- Built-in metadata filtering

**How it works**: Uses approximate nearest neighbor (ANN) algorithms to quickly find similar vectors

#### 3. **Sentence Transformers** - Text Embedding
```python
from sentence_transformers import SentenceTransformer
```
**What it does**: Converts text into numerical vectors (embeddings)
**Why we use it**:
- Captures semantic meaning ("car" and "automobile" have similar embeddings)
- Pre-trained on large datasets
- Optimized for sentence-level understanding

**How it works**: Uses transformer neural networks to encode text into high-dimensional vectors

#### 4. **PyPDF2 & python-docx** - Document Processing
```python
import PyPDF2
from docx import Document
```
**What they do**: Extract text from different file formats
**Why we use them**: Each format requires specialized parsing to extract clean text

### Code Architecture Deep Dive 🏗️

#### 1. Document Processing Pipeline
```python
def load_documents(self, folder_path):
    # Step 1: Find all supported files
    # Step 2: Extract text based on file type
    # Step 3: Split into chunks with overlap
    # Step 4: Create embeddings for each chunk
    # Step 5: Store in vector database
```

**Why chunking?**
- **Context Window Limits**: LLMs have maximum input lengths
- **Relevance**: Smaller chunks = more precise retrieval
- **Overlap**: Ensures important information isn't split across boundaries

#### 2. Embedding Generation Process
```python
def create_embeddings(self, texts):
    # Convert text chunks to numerical vectors
    embeddings = self.embedding_model.encode(texts)
    return embeddings
```

**Why embeddings work**:
- Similar concepts cluster together in vector space
- Mathematical operations can measure semantic similarity
- Enables fast search through millions of documents

#### 3. Retrieval Mechanism
```python
def retrieve_relevant_chunks(self, query, n_results=5):
    # Step 1: Convert query to embedding
    query_embedding = self.embedding_model.encode([query])
    
    # Step 2: Search vector database
    results = self.collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )
    return results
```

**How similarity search works**:
1. Query gets converted to same vector space as documents
2. Cosine similarity calculated between query and all document vectors
3. Top-k most similar chunks retrieved
4. Similarity scores help rank relevance

#### 4. Response Generation with Streaming
```python
def generate_response_streaming(self, query, context):
    # Combine retrieved context with user query
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    
    # Generate response using local LLM with streaming
    for token in self.llm(prompt, stream=True, max_tokens=self.max_tokens):
        token_text = token['choices'][0]['text']
        print(token_text, end='', flush=True)
        yield token_text
```

**Streaming Benefits:**
- **Immediate Feedback**: Users see responses as they're generated
- **Perceived Performance**: Feels faster even if total time is the same
- **Interruptible**: Can stop generation early if needed
- **Progressive Disclosure**: Information appears progressively

### Why This Architecture Works 🎯

#### 1. **Separation of Concerns**
- **Retrieval System**: Handles finding relevant information
- **Generation System**: Focuses on creating coherent responses
- **Storage System**: Manages persistent data efficiently

#### 2. **Scalability**
- Vector databases scale to millions of documents
- Chunking allows processing of large documents
- Local models avoid API rate limits

#### 3. **Accuracy**
- Grounding in actual documents reduces hallucinations
- Semantic search finds conceptually related content
- Context-aware generation produces relevant answers

#### 4. **Privacy & Control**
- All processing happens locally
- No data sent to external services
- Full control over model behavior and responses
- Real-time streaming without external dependencies

### Learning Exercises 📝

#### Beginner Level
1. **Experiment with streaming**: Compare streaming vs non-streaming modes
2. **Experiment with chunk sizes**: Try different values and see how it affects retrieval quality
3. **Compare embedding models**: Test different sentence transformer models
4. **Analyze similarity scores**: Look at the relevance scores returned by ChromaDB

#### Intermediate Level
1. **Optimize streaming performance**: Experiment with different model sizes and streaming settings
2. **Implement custom document loaders**: Add support for new file formats
3. **Experiment with retrieval strategies**: Try different numbers of retrieved chunks
4. **Customize chat templates**: Create templates for specific use cases

#### Advanced Level
1. **Implement streaming optimizations**: Add token buffering and smart chunking
2. **Implement hybrid search**: Combine semantic and keyword search
3. **Add re-ranking**: Use cross-encoders to re-rank retrieved results
4. **Optimize performance**: Profile and optimize the embedding and retrieval pipeline

### Common Pitfalls and Solutions 🚨

#### 1. **Poor Retrieval Quality**
**Problem**: Irrelevant chunks being retrieved
**Solutions**:
- Adjust chunk size and overlap
- Try different embedding models
- Implement query expansion or reformulation

#### 2. **Slow Performance**
**Problem**: Long response times
**Solutions**:
- Use quantized models (Q4_K_M, Q5_K_M)
- Reduce context window size
- Implement caching for frequent queries

#### 3. **Memory Issues**
**Problem**: Out of memory errors
**Solutions**:
- Use smaller models
- Process documents in batches
- Enable streaming to reduce memory pressure
- Use quantized models for better memory efficiency

#### 4. **Streaming Performance Issues**
**Problem**: Slow or choppy streaming output
**Solutions**:
- Use faster models (Q4_K_M quantization)
- Reduce context window size
- Optimize token generation settings
- Ensure adequate system resources

### Further Learning Resources 📖

1. **Streaming Technologies**: Learn about WebSockets, Server-Sent Events, and real-time communication
2. **Vector Databases**: Learn about Pinecone, Weaviate, and Qdrant
3. **Embedding Models**: Explore BGE, E5, and other state-of-the-art models
4. **LLM Optimization**: Study quantization, pruning, and distillation techniques
5. **RAG Improvements**: Research advanced techniques like HyDE, RAG-Fusion, and RAPTOR
6. **Performance Optimization**: Study token generation optimization and streaming architectures

This architecture demonstrates fundamental concepts in modern AI applications: vector search, semantic understanding, local AI deployment, and real-time streaming. Understanding these concepts will help you build more sophisticated AI systems!

## Troubleshooting 🔧

### Common Issues

#### Streaming Output Issues
- **Problem**: Streaming not working or responses appear all at once
  - **Solution**: Verify streaming is enabled with `--streaming` flag
  - **Check**: Ensure `advanced.experimental.streaming: true` in config.yaml
  - **Debug**: Use `help` command to check current streaming status

- **Problem**: Streaming too slow or choppy
  - **Solution**: Reduce model size or context window
  - **Alternative**: Use quantized models (Q4_K_M) for faster token generation

#### GGUF Model Loading Issues
- **Error**: "Failed to load GGUF model"
  - **Solution**: Ensure the model file exists and is a valid GGUF format
  - **Check**: Verify the model path is correct and the file isn't corrupted
  - **Debug**: Use `--verbose` flag for detailed error messages

#### Chat Template Detection
- **Error**: "Could not detect chat template"
  - **Solution**: Manually specify template with `--chat_template` parameter
  - **Options**: `chatml`, `llama2`, `alpaca`, or `auto` for auto-detection

#### Memory Issues
- **Error**: "Out of memory" or slow performance
  - **Solution**: Reduce `n_ctx` parameter or use a smaller model
  - **Alternative**: Enable GPU acceleration if available

#### Document Processing
- **Error**: "No documents found" or "Failed to process documents"
  - **Solution**: Check folder path and ensure supported file formats
  - **Verify**: Documents folder contains .pdf, .docx, .txt, or .md files

### Performance Optimization
- **Streaming Performance**: Enable streaming for real-time feedback
- **GPU Acceleration**: Use GPU when available for faster token generation
- **Model Selection**: Use quantized models (Q4_K_M, Q5_K_M) for better performance
- **Context Management**: Reduce context window for faster inference
- **Document Processing**: Adjust chunk size based on document types
- **Memory Management**: Monitor memory usage with streaming enabled

## Development 👨‍💻

### Project Structure
```
DocuChat/
├── docuchat.py          # Main application with streaming support
├── config/
│   └── config.yaml      # Configuration including streaming settings
├── documents/           # Place your documents here
│   └── .gitkeep
├── models/              # Place your GGUF models here
│   └── .gitkeep
├── embeddings/          # Downloaded embedding models
│   ├── .gitkeep
│   └── README.md
├── tests/               # Test suite
│   ├── test_docuchat.py
│   └── test_embedding_models.py
├── requirements.txt     # Python dependencies
├── README.md           # This comprehensive documentation
└── .gitignore          # Git ignore patterns
```

### Testing
```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_chat_templates.py
python -m pytest tests/test_docuchat.py

# Test GGUF auto-detection
python test_auto_detection.py
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## Roadmap 🗺️

### Upcoming Features
- **Enhanced GGUF Support**: Additional metadata extraction and validation
- **Multi-modal Support**: Image and audio document processing
- **Advanced RAG**: Hybrid search and re-ranking capabilities
- **Web Interface**: Browser-based chat interface
- **API Endpoints**: RESTful API for integration
- **Cloud Deployment**: Docker containers and cloud deployment guides

### Performance Improvements
- **Enhanced Streaming**: Improved character-by-character response generation
- **Streaming Controls**: Fine-grained streaming configuration options
- **Caching System**: Intelligent caching for faster responses
- **Parallel Processing**: Multi-threaded document processing
- **Memory Optimization**: Reduced memory footprint with streaming

## License 📄

This project is licensed under the MIT License - see the LICENSE file for details.

## Support 🤝

- **Issues**: Report bugs and request features on GitHub Issues
- **Discussions**: Join community discussions on GitHub Discussions
- **Documentation**: Check the docs/ folder for detailed guides

## Acknowledgments 🙏

- **llama-cpp-python**: For excellent GGUF model support
- **ChromaDB**: For efficient vector storage and retrieval
- **Sentence Transformers**: For high-quality embeddings
- **Hugging Face**: For the amazing model ecosystem

---

**Happy chatting with your documents! 🚀📚**