# DocuChat 🤖📚

A powerful RAG (Retrieval-Augmented Generation) application that lets you chat with your documents using local language models. DocuChat processes your documents, creates vector embeddings, and provides intelligent responses based on your document content.

## Features ✨

- **Local LLM Support**: Uses GGUF format models via llama-cpp-python
- **Multiple Document Formats**: Supports PDF, DOCX, TXT, and Markdown files
- **Vector Search**: Efficient document retrieval using ChromaDB and sentence transformers
- **Interactive Chat**: Real-time conversation interface
- **Flexible Configuration**: Support for both YAML configuration files and command-line arguments
- **Customizable**: Multiple chat templates and configurable parameters
- **Performance Optimized**: Chunking, caching, and efficient embedding generation

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

### Configuration Structure

The YAML configuration file supports the following sections:

- **model**: LLM settings (path, context length, sampling parameters)
- **documents**: Document processing settings (folder path, chunking)
- **embeddings**: Embedding model configuration
- **vector_store**: ChromaDB settings
- **rag**: Retrieval-augmented generation parameters
- **ui**: User interface and chat settings
- **logging**: Logging configuration
- **performance**: Performance optimization settings

### Mixed Configuration Example

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

### Other Options
- `--embedding_model`: Sentence transformer model (default: all-MiniLM-L6-v2)
- `--n_retrieve`: Number of documents to retrieve (default: 5)
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

1. **Document Processing**: Documents are loaded and split into overlapping chunks
2. **Embedding Generation**: Text chunks are converted to vector embeddings using sentence transformers
3. **Vector Storage**: Embeddings are stored in ChromaDB for efficient similarity search
4. **Query Processing**: User queries are embedded and matched against document chunks
5. **Response Generation**: Retrieved context is provided to the LLM for generating responses

## Configuration Priority 🎯

Settings are applied in the following order (later overrides earlier):

1. Default values in `DocuChatConfig`
2. YAML configuration file settings
3. Command-line argument overrides

This allows for flexible configuration management where you can:
- Set common settings in YAML files
- Override specific settings via command line
- Use different config files for different scenarios

## Performance Tips 🚀

- **GPU Acceleration**: Use `--n_gpu_layers` to offload layers to GPU
- **Memory Management**: Adjust `--n_ctx` based on available RAM
- **Chunk Size**: Larger chunks provide more context but slower retrieval
- **Embedding Model**: Balance between speed and quality:
  - Fast: `all-MiniLM-L6-v2`
  - Better: `all-mpnet-base-v2`
  - Q&A optimized: `multi-qa-MiniLM-L6-cos-v1`

## Troubleshooting 🔍

### Common Issues

1. **Model not found**: Ensure the model path is correct and the file exists
2. **Out of memory**: Reduce `n_ctx` or use a smaller model
3. **Slow performance**: Enable GPU layers or use a smaller embedding model
4. **No documents loaded**: Check folder path and file permissions

### Debug Mode

```bash
python docuchat.py --verbose
```

## Contributing 🤝

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License 📜

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments 🙏

- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) for local LLM inference
- [ChromaDB](https://github.com/chroma-core/chroma) for vector storage
- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) for embeddings
- The open-source AI community for making this possible
