# DocuChat 🤖📚

A powerful RAG (Retrieval-Augmented Generation) application that lets you chat with your documents using local language models. DocuChat processes your documents, creates vector embeddings, and provides intelligent responses based on your document content.

## Recent Updates 🆕

### Version 2.0 - Enhanced GGUF Auto-Detection
- **🎯 Robust GGUF Processing**: Improved metadata reading with fallback mechanisms
- **🔧 Enhanced Error Handling**: Better diagnostics for model loading issues
- **📊 Advanced Chat Templates**: Auto-detecting chat templates with comprehensive format support
- **🚀 Performance Optimizations**: Faster document processing and embedding generation
- **🧪 Comprehensive Testing**: New test suite for GGUF functionality and chat templates

## Features ✨

### Advanced AI Capabilities
- **🎯 Auto-Detecting Chat Templates**: Automatically detects and applies the correct chat template format
- **🔧 Robust GGUF Processing**: Enhanced metadata reading with comprehensive error handling
- **🤖 Local LLM Support**: Uses GGUF format models via llama-cpp-python
- **📚 Multiple Document Formats**: Supports PDF, DOCX, TXT, and Markdown files
- **🔍 Vector Search**: Efficient document retrieval using ChromaDB and sentence transformers
- **💬 Interactive Chat**: Real-time conversation interface
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
- `--n_retrieve`: Number of documents to retrieve (default: 5)
- `--no-rag`: Disable RAG (Retrieval-Augmented Generation) and use LLM only

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

### Advanced Usage
```bash
# High-performance setup with custom settings
python docuchat.py \
  --model_path ./models/llama-2-13b.gguf \
  --folder_path ./research_papers \
  --chunk_size 1500 \
  --chunk_overlap 300 \
  --n_retrieve 8 \
  --temperature 0.3 \
  --max_tokens 4096 \
  --verbose
```

## Interactive Commands 💬

Once in interactive mode, you can use these commands:
- `help` - Show available commands
- `quit`, `exit`, `bye` - Exit the application
- Any other text - Chat with your documents

## Supported File Formats 📄
- **PDF** (.pdf) - Extracted using PyPDF2
- **Word Documents** (.docx, .doc) - Processed with python-docx
- **Text Files** (.txt) - Plain text
- **Markdown** (.md) - Markdown formatted text

## How It Works 🔧

### RAG Mode (Default)
1. **Document Processing**: Documents are loaded and split into overlapping chunks
2. **Embedding Generation**: Text chunks are converted to vector embeddings using sentence transformers
3. **Vector Storage**: Embeddings are stored in ChromaDB for efficient similarity search
4. **Query Processing**: User queries are embedded and matched against document chunks
5. **Context Retrieval**: Most relevant document chunks are retrieved based on similarity
6. **Response Generation**: Retrieved context is combined with the query and sent to the LLM
7. **Chat Template Application**: Responses are formatted using the appropriate chat template

### LLM-Only Mode (--no-rag)
When using the `--no-rag` flag, DocuChat bypasses the document retrieval process and sends queries directly to the language model, functioning as a standard chatbot without document context.

## Troubleshooting 🔧

### Common Issues

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
- Use GPU acceleration when available
- Adjust chunk size based on document types
- Reduce context window for faster inference
- Use quantized models (Q4_K_M, Q5_K_M) for better performance

## Development 👨‍💻

### Project Structure
```
DocuChat/
├── docuchat.py          # Main application
├── config/              # Configuration files
├── documents/           # Place your documents here
├── models/              # Place your GGUF models here
├── tests/               # Test suite
├── examples/            # Usage examples
└── docs/                # Documentation
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
- **Streaming Responses**: Real-time response generation
- **Caching System**: Intelligent caching for faster responses
- **Parallel Processing**: Multi-threaded document processing
- **Memory Optimization**: Reduced memory footprint

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