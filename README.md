# DocuChat 🤖📚

A powerful RAG (Retrieval-Augmented Generation) application that lets you chat with your documents using local language models. DocuChat processes your documents, creates vector embeddings, and provides intelligent responses based on your document content with real-time streaming output.

## ✨ Key Features

### 🚀 Advanced AI Capabilities
- **⚡ Real-time Streaming**: Character-by-character response generation for immediate feedback
- **🎯 Auto-Detecting Chat Templates**: Automatically detects and applies the correct chat template format
- **🤖 Local LLM Support**: Uses GGUF format models via llama-cpp-python for privacy and cost efficiency
- **📚 Multiple Document Formats**: Supports PDF, DOCX, TXT, and Markdown files
- **🔍 Semantic Vector Search**: Efficient document retrieval using ChromaDB and sentence transformers
- **💬 Interactive Chat**: Real-time conversation interface
- **⚙️ Flexible Configuration**: Support for both YAML configuration files and command-line arguments
- **🎨 Customizable Templates**: Multiple chat templates (ChatML, Llama2, Alpaca) with auto-detection


### 🔧 Technical Excellence
- **📊 Persistent Vector Database**: ChromaDB storage for fast startup and efficient similarity search
- **🔄 Message-Based Architecture**: Modern chat completion API using llama-cpp-python
- **⚡ Performance Optimized**: Chunking, caching, and efficient embedding generation
- **🛡️ Privacy-First**: All processing happens locally, no data sent to external services
- **💾 Smart Caching**: Vector embeddings persist between sessions

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- At least 8GB RAM (16GB+ recommended for larger models)
- CUDA-compatible GPU (optional, for GPU acceleration)

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/DocuChat.git
cd DocuChat

# Install dependencies
pip install -r requirements.txt
```

### Download a Model
Download a GGUF format model from Hugging Face:
```bash
# Example: Download a Llama 2 7B model
wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf
```
Place the model in the `models/` directory.

### Basic Usage
```bash
# Using default configuration
python docuchat.py --model_path ./models/your-model.gguf --folder_path ./documents

# Interactive chat with streaming
python docuchat.py --model_path ./models/your-model.gguf --folder_path ./documents --streaming

# Single query mode
python docuchat.py --model_path ./models/your-model.gguf --query "What is the main topic?"
```

## 📋 Configuration

DocuChat supports flexible configuration through YAML files and command-line arguments, with CLI arguments taking precedence.

### YAML Configuration
Create a `config/config.yaml` file:
```yaml
model:
  model_path: "./models/llama-2-7b-chat.Q4_K_M.gguf"
  n_ctx: 4096
  temperature: 0.7
  max_tokens: 2048

documents:
  folder_path: "./documents"
  chunk_size: 1000
  chunk_overlap: 200

embeddings:
  model: "all-MiniLM-L6-v2"

rag:
  n_retrieve: 5
  enabled: true

ui:
  streaming: true
  system_prompt: "You are a helpful assistant."
  chat_template: "auto"

logging:
  verbose: false
```

### Command Line Options

#### Core Options
- `--config`: Path to YAML configuration file (default: config/config.yaml)
- `--model_path`: Path to GGUF model file (required)
- `--folder_path`: Path to documents folder
- `--query`: Single query mode (non-interactive)

#### Model Configuration
- `--n_ctx`: Context window size (default: 4096)
- `--temperature`: Sampling temperature (default: 0.7)
- `--max_tokens`: Maximum tokens to generate (default: 2048)
- `--chat_template`: Chat template format (auto, chatml, llama2, alpaca)

#### Document Processing
- `--chunk_size`: Document chunk size (default: 1000)
- `--chunk_overlap`: Chunk overlap size (default: 200)

#### Embedding Models
- `--embedding_model`: Sentence transformer model (default: all-MiniLM-L6-v2)
- `--download_embedding_model`: Download model from Hugging Face to embeddings/ folder

#### RAG Configuration
- `--n_retrieve`: Number of documents to retrieve (default: 5)
- `--no-rag`: Disable RAG and use LLM only

#### Streaming & Display
- `--streaming`: Enable streaming output (default: enabled)
- `--no-streaming`: Disable streaming output


#### Other Options
- `--system_prompt`: Custom system prompt
- `--verbose`: Enable verbose logging

## 💡 Usage Examples

### Basic RAG Chat
```bash
# Start interactive chat with documents
python docuchat.py --model_path ./models/llama-2-7b.gguf --folder_path ./documents
```

### Configuration File Usage
```bash
# Use default config file
python docuchat.py

# Use custom config file
python docuchat.py --config my_config.yaml

# Override config with CLI arguments
python docuchat.py --model_path ./custom-model.gguf --temperature 0.9
```

### Embedding Model Management
```bash
# Download a custom embedding model
python docuchat.py --download_embedding_model "sentence-transformers/all-mpnet-base-v2"

# Use downloaded model
python docuchat.py --embedding_model "sentence-transformers--all-mpnet-base-v2"

# Use different Hugging Face model directly
python docuchat.py --embedding_model "multi-qa-MiniLM-L6-cos-v1"
```

### Advanced Usage
```bash
# High-performance setup with streaming
python docuchat.py \
  --model_path ./models/llama-2-13b.gguf \
  --folder_path ./research_papers \
  --chunk_size 1500 \
  --chunk_overlap 300 \
  --n_retrieve 8 \
  --temperature 0.3 \
  --streaming \
  --verbose

# LLM-only mode (no document retrieval)
python docuchat.py --no-rag --model_path ./models/llama-2-7b.gguf

# Single query mode
python docuchat.py \
  --model_path ./models/llama-2-7b.gguf \
  --query "Explain quantum computing"
```

## 🎮 Interactive Commands

Once in interactive mode:
- `help` - Show available commands and current configuration
- `quit`, `exit`, `bye` - Exit the application
- Any other text - Chat with your documents

### Status Information
The application displays:
- **Model**: Currently loaded GGUF model
- **Documents**: Number of processed documents and chunks
- **Embedding Model**: Active embedding model
- **Streaming**: Current streaming status
- **RAG Mode**: Whether document retrieval is active


## 🧠 Embedding Models

### Recommended Models
- `all-MiniLM-L6-v2` (default) - Fast, good quality, 384 dimensions
- `all-mpnet-base-v2` - Slower, better quality, 768 dimensions
- `multi-qa-MiniLM-L6-cos-v1` - Optimized for Q&A tasks
- `paraphrase-multilingual-MiniLM-L12-v2` - Multilingual support

### Model Selection Guidelines
- **Speed Priority**: Use `all-MiniLM-L6-v2`
- **Quality Priority**: Use `all-mpnet-base-v2`
- **Q&A Tasks**: Use `multi-qa-MiniLM-L6-cos-v1`
- **Multilingual**: Use `paraphrase-multilingual-MiniLM-L12-v2`
- **Offline Use**: Download models locally with `--download_embedding_model`

## 📄 Supported File Formats
- **PDF** (.pdf) - Extracted using PyPDF2
- **Word Documents** (.docx, .doc) - Processed with python-docx
- **Text Files** (.txt) - Plain text
- **Markdown** (.md) - Markdown formatted text

## 🔧 How It Works

### RAG Mode (Default)
1. **Document Processing**: Documents are loaded and split into overlapping chunks
2. **Embedding Generation**: Text chunks are converted to vector embeddings
3. **Vector Storage**: Embeddings are stored in ChromaDB for efficient similarity search
4. **Query Processing**: User queries are embedded and matched against document chunks
5. **Context Retrieval**: Most relevant document chunks are retrieved
6. **Response Generation**: Retrieved context is combined with the query and sent to the LLM
7. **Streaming Output**: Responses are generated token-by-token in real-time

### LLM-Only Mode (--no-rag)
Bypasses document retrieval and sends queries directly to the language model.

### Streaming Implementation
- **Real-time Output**: Tokens displayed as they're generated
- **Better UX**: Immediate feedback without waiting
- **Interruptible**: Generation can be stopped mid-stream
- **Configurable**: Enable/disable via CLI or config file

### Vector Database Storage
The application uses ChromaDB to persistently store vector embeddings:
- **Storage Location**: `./chroma_db` directory
- **Persistence**: Database persists between sessions
- **Benefits**: Faster startup, efficient similarity search, reduced computational overhead

## 🏗️ Project Structure
```
DocuChat/
├── docuchat.py          # Main application
├── config/
│   └── config.yaml      # Configuration file
├── documents/           # Place your documents here
│   └── .gitkeep
├── models/              # Place your GGUF models here
│   └── .gitkeep
├── embeddings/          # Downloaded embedding models
│   ├── .gitkeep
│   └── README.md
├── requirements.txt     # Python dependencies
├── README.md           # This documentation
├── .gitignore          # Git ignore patterns
├── run_no_rag.bat      # Windows batch file for LLM-only mode
└── run_with_rag.bat    # Windows batch file for RAG mode
```

## 🔧 Troubleshooting

### Common Issues

#### Model Loading
- **Error**: "Failed to load GGUF model"
  - **Solution**: Verify model file exists and is valid GGUF format
  - **Check**: Ensure correct model path and file isn't corrupted

#### Streaming Issues
- **Problem**: Streaming not working
  - **Solution**: Use `--streaming` flag or set `ui.streaming: true` in config
  - **Check**: Verify streaming status with `help` command

#### Memory Issues
- **Error**: "Out of memory"
  - **Solution**: Reduce `n_ctx` parameter or use smaller model
  - **Alternative**: Use quantized models (Q4_K_M, Q5_K_M)

#### Document Processing
- **Error**: "No documents found"
  - **Solution**: Check folder path and file formats
  - **Verify**: Ensure supported file types (.pdf, .docx, .txt, .md)

### Performance Optimization
- **Model Selection**: Use quantized models for better performance
- **Context Management**: Reduce context window for faster inference
- **Streaming**: Enable streaming for real-time feedback
- **GPU Acceleration**: Use GPU when available
- **Memory Management**: Monitor usage with verbose logging

## 🛠️ Development

### Dependencies
Core dependencies are listed in `requirements.txt`:
- `llama-cpp-python>=0.2.0` - Local LLM inference
- `chromadb>=0.4.0` - Vector database
- `sentence-transformers>=2.2.0` - Text embeddings
- `PyPDF2>=3.0.0` - PDF processing
- `python-docx>=0.8.11` - Word document processing
- `pyyaml>=6.0.0` - Configuration files
- `colorama>=0.4.4` - Terminal colors

### Architecture
The application follows a modular architecture:
- **DocumentProcessor**: Handles file loading and text chunking
- **VectorStore**: Manages embeddings and similarity search
- **DocuChatConfig**: Configuration management
- **DocuChat**: Main application logic with streaming support

### Key Features Implementation
- **Message-based Architecture**: Uses llama-cpp-python's chat completion API
- **Streaming Support**: Real-time token generation with configurable output

- **Template Auto-detection**: Automatic chat template format detection
- **Flexible Configuration**: YAML and CLI argument support with precedence

## 🗺️ Roadmap

### Upcoming Features
- **Web Interface**: Browser-based chat interface
- **API Endpoints**: RESTful API for integration
- **Multi-modal Support**: Image and audio document processing
- **Advanced RAG**: Hybrid search and re-ranking capabilities
- **Cloud Deployment**: Docker containers and deployment guides

### Performance Improvements
- **Enhanced Streaming**: Improved real-time response generation
- **Caching System**: Intelligent caching for faster responses
- **Parallel Processing**: Multi-threaded document processing
- **Memory Optimization**: Reduced memory footprint

## 📄 License

This project is licensed under the MIT License.

## 🤝 Support

- **Issues**: Report bugs and request features on GitHub Issues
- **Documentation**: Check this README for comprehensive guides
- **Community**: Join discussions for help and feature requests

## 🙏 Acknowledgments

- **llama-cpp-python**: Excellent GGUF model support and chat completion API
- **ChromaDB**: Efficient vector storage and retrieval
- **Sentence Transformers**: High-quality text embeddings
- **Hugging Face**: Amazing model ecosystem and community

---

**Happy chatting with your documents! 🚀📚**
