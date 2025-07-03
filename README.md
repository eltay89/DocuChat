# DocuChat 🤖📚

**DocuChat** is a sophisticated, privacy-first RAG (Retrieval-Augmented Generation) application that enables intelligent conversations with your documents using local language models. Built with Python and powered by cutting-edge AI technologies, DocuChat transforms your document collections into an interactive knowledge base that you can query naturally.

## 🎯 What is DocuChat?

DocuChat is a local AI-powered document chat system that:
- **Processes your documents** (PDF, DOCX, TXT, Markdown) into searchable knowledge
- **Uses vector embeddings** to understand document content semantically
- **Leverages local LLMs** (Large Language Models) for intelligent responses
- **Maintains complete privacy** - all processing happens on your machine
- **Provides real-time streaming** responses for immediate feedback
- **Remembers context** across conversations within a session

### How It Works (Simple Explanation)
1. **📄 Document Ingestion**: You point DocuChat to a folder containing your documents
2. **🔍 Content Analysis**: Documents are broken into chunks and converted to mathematical vectors (embeddings)
3. **💾 Knowledge Storage**: These vectors are stored in a local database for fast retrieval
4. **❓ Query Processing**: When you ask a question, DocuChat finds the most relevant document chunks
5. **🤖 AI Response**: A local language model uses the relevant content to generate accurate, contextual answers
6. **⚡ Real-time Output**: Responses stream to you character-by-character as they're generated

## ✨ Key Features

### 🚀 Advanced AI Capabilities
- **⚡ Real-time Streaming**: Watch responses generate character-by-character for immediate feedback, just like ChatGPT
- **🎯 Auto-Detecting Chat Templates**: Automatically detects and applies the correct conversation format for your specific model (ChatML, Llama2, Alpaca, etc.)
- **🤖 Local LLM Support**: Run powerful language models entirely on your machine using efficient GGUF format - no internet required, no API costs
- **📚 Multiple Document Formats**: Seamlessly process PDF documents, Word files (.docx), plain text (.txt), and Markdown (.md) files
- **🔍 Semantic Vector Search**: Advanced similarity search using ChromaDB and sentence transformers - finds relevant content even when exact keywords don't match
- **💬 Interactive Chat Interface**: Natural conversation flow with context awareness and command support
- **⚙️ Flexible Configuration**: Configure everything via YAML files or command-line arguments with intelligent defaults
- **🎨 Smart Model Integration**: Supports various model architectures with automatic optimization and template detection
- **🛡️ Robust Error Handling**: Comprehensive error recovery, timeout protection, and graceful degradation
- **📊 Source Attribution**: See exactly which documents contributed to each response for transparency and verification

### 🔧 Technical Excellence
- **📊 Persistent Vector Database**: ChromaDB storage ensures fast startup times and efficient similarity search across sessions
- **🔄 Modern Architecture**: Built on llama-cpp-python's chat completion API for optimal performance and compatibility
- **⚡ Performance Optimized**: Intelligent text chunking, embedding caching, and memory management for smooth operation
- **🛡️ Privacy-First Design**: Zero external API calls - your documents and conversations never leave your machine
- **💾 Smart Caching System**: Vector embeddings and metadata persist between sessions, eliminating redundant processing
- **🔒 Security Focused**: Uses SHA-256 hashing, JSON serialization (not pickle), and secure file handling practices
- **📝 Type-Safe Code**: Comprehensive type hints throughout the codebase for better maintainability and IDE support

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+**: Modern Python installation (3.9+ recommended)
- **Memory**: At least 8GB RAM (16GB+ recommended for larger models)
- **Storage**: 5-50GB free space (depending on model size)
- **GPU** (Optional): CUDA-compatible GPU for faster inference
- **Operating System**: Windows, macOS, or Linux

### 📦 Installation

#### Step 1: Clone the Repository
```bash
# Clone the repository
git clone https://github.com/yourusername/DocuChat.git
cd DocuChat

# Create a virtual environment (recommended)
python -m venv docuchat-env

# Activate virtual environment
# On Windows:
docuchat-env\Scripts\activate
# On macOS/Linux:
source docuchat-env/bin/activate
```

#### Step 2: Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# For GPU support (optional, if you have CUDA)
pip install llama-cpp-python[cuda]
```

### 🤖 Download a Language Model

DocuChat uses GGUF format models. Here are some recommended options:

#### Quick Start Models (Smaller, Faster)
```bash
# Llama 3.1 8B (Recommended for most users)
wget https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

# Phi-3 Mini (Lightweight, good for testing)
wget https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf
```

#### High-Quality Models (Larger, Better Results)
```bash
# Llama 3.1 70B (Best quality, requires 32GB+ RAM)
wget https://huggingface.co/bartowski/Meta-Llama-3.1-70B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf

# Mixtral 8x7B (Good balance of quality and speed)
wget https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf
```

**Place downloaded models in the `models/` directory.**

### 📄 Prepare Your Documents

1. Create a `documents/` folder (or use existing one)
2. Add your files: PDF, DOCX, TXT, or Markdown
3. Organize in subfolders if desired (DocuChat will scan recursively)

```
documents/
├── research_papers/
│   ├── paper1.pdf
│   └── paper2.pdf
├── manuals/
│   ├── user_guide.docx
│   └── technical_specs.md
└── notes.txt
```

### 🎯 Basic Usage

#### Interactive Chat Mode (Recommended)
```bash
# Start chatting with your documents
python docuchat.py --model_path ./models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --folder_path ./documents

# With verbose output to see what's happening
python docuchat.py --model_path ./models/your-model.gguf --folder_path ./documents --verbose

# Using a configuration file
python docuchat.py --config config/config.yaml
```

#### Single Query Mode
```bash
# Ask a specific question without entering interactive mode
python docuchat.py --model_path ./models/your-model.gguf --folder_path ./documents --query "What are the main findings in the research papers?"

# LLM-only mode (no document retrieval)
python docuchat.py --model_path ./models/your-model.gguf --no-rag --query "Explain quantum computing"
```

#### First Run Example
```bash
# Your first command might look like this:
python docuchat.py \
  --model_path ./models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  --folder_path ./documents \
  --verbose \
  --streaming

# This will:
# 1. Load your model
# 2. Process all documents in ./documents/
# 3. Create vector embeddings
# 4. Start interactive chat with streaming responses
# 5. Show detailed progress information
```

## ⚙️ Configuration

DocuChat offers flexible configuration through both YAML files and command-line arguments, allowing you to customize every aspect of the application.

### 📝 YAML Configuration

The primary configuration file is located at `config/config.yaml`. This file controls all aspects of DocuChat's behavior and is organized into logical sections:

```yaml
# Language Model Configuration
model:
  # Path to your GGUF model file (required)
  model_path: "./models/llama-2-7b-chat.Q4_K_M.gguf"
  
  # Maximum context window size in tokens
  n_ctx: 4096
  
  # Controls randomness (0.0-1.0, higher = more random)
  temperature: 0.7
  
  # Maximum number of tokens to generate per response
  max_tokens: 2048

# Document Processing Settings
documents:
  # Directory containing documents to process
  folder_path: "./documents"
  
  # Size of each document chunk in characters
  chunk_size: 1000
  
  # Overlap between consecutive chunks in characters
  chunk_overlap: 200

# Embedding Model Configuration
embeddings:
  # Sentence-transformers model name
  model: "all-MiniLM-L6-v2"

# Retrieval-Augmented Generation Settings
rag:
  # Number of document chunks to retrieve per query
  n_retrieve: 5
  
  # Enable/disable RAG functionality
  enabled: true

# User Interface Configuration
ui:
  # Enable token-by-token streaming
  streaming: true
  
  # System prompt for the AI assistant
  system_prompt: "You are a helpful assistant."
  
  # Chat template format (auto, chatml, llama2, alpaca)
  chat_template: "auto"

# Logging Configuration
logging:
  # Enable verbose logging
  verbose: false
```

### 🖥️ Command-Line Arguments

DocuChat provides extensive command-line options that override the YAML configuration. This allows for quick experimentation without editing the config file.

#### Basic Usage

```bash
python docuchat.py --model_path models/llama-2-7b-chat.Q4_K_M.gguf --folder_path docs
```

#### Complete Command-Line Reference

##### Core Options
- `--config`: Path to YAML configuration file (default: config/config.yaml)
- `--model_path`: Path to GGUF model file (required)
- `--folder_path`: Path to documents folder
- `--query`: Single query mode (non-interactive)

##### Model Configuration
- `--n_ctx`: Context window size (default: 4096)
- `--temperature`: Sampling temperature (default: 0.7)
- `--max_tokens`: Maximum tokens to generate (default: 2048)
- `--chat_template`: Chat template format (auto, chatml, llama2, alpaca)

##### Document Processing
- `--chunk_size`: Document chunk size (default: 1000)
- `--chunk_overlap`: Chunk overlap size (default: 200)

##### Embedding Models
- `--embedding_model`: Sentence transformer model (default: all-MiniLM-L6-v2)
- `--download_embedding_model`: Download model from Hugging Face to embeddings/ folder

##### RAG Configuration
- `--n_retrieve`: Number of documents to retrieve (default: 5)
- `--no-rag`: Disable RAG and use LLM only

##### Streaming & Display
- `--streaming`: Enable streaming output (default: enabled)
- `--no-streaming`: Disable streaming output

##### Other Options
- `--system_prompt`: Custom system prompt
- `--verbose`: Enable verbose logging

### 🔄 Configuration Precedence

DocuChat follows a clear precedence order when determining configuration values:

1. **Command-line arguments** (highest priority)
2. **Custom config file** (if specified with `--config`)
3. **Default config file** (`config/config.yaml`)
4. **Hardcoded defaults** (lowest priority)

### 🧩 Advanced Configuration

#### Multiple Configuration Profiles

You can maintain multiple configuration files for different use cases:

```bash
# Technical documentation analysis
python docuchat.py --config configs/technical.yaml

# Creative writing assistant
python docuchat.py --config configs/creative.yaml
```

#### Configuration Validation

DocuChat validates all configuration values at startup and provides helpful error messages for invalid settings. This prevents runtime errors due to misconfiguration.

## 💡 Usage Examples

DocuChat is designed to be versatile and adaptable to various use cases. Here are comprehensive examples to help you get the most out of the application.

### 🤖 Interactive Chat Mode

**Basic Interactive Session**
```bash
python docuchat.py --model_path models/llama-2-7b-chat.Q4_K_M.gguf --folder_path ./my_documents
```

**With Custom Configuration**
```bash
python docuchat.py --config my_custom_config.yaml
```

**With Adjusted Parameters**
```bash
python docuchat.py --model_path models/mistral-7b-instruct.Q4_K_M.gguf \
                  --folder_path ./technical_docs \
                  --temperature 0.5 \
                  --n_retrieve 8 \
                  --system_prompt "You are a technical expert who provides detailed explanations."
```

### 📝 Single Query Mode

**Basic Query**
```bash
python docuchat.py --model_path models/llama-2-7b-chat.Q4_K_M.gguf \
                  --folder_path ./my_documents \
                  --query "What are the key features of the product?"
```

**Complex Query with Output Redirection**
```bash
python docuchat.py --model_path models/llama-2-7b-chat.Q4_K_M.gguf \
                  --folder_path ./financial_reports \
                  --query "Summarize the Q2 financial performance and highlight key metrics" \
                  --no-streaming > financial_summary.txt
```

**Batch Processing Multiple Queries**
```bash
# Create a file with one query per line
cat queries.txt | while read query; do
  python docuchat.py --model_path models/llama-2-7b-chat.Q4_K_M.gguf \
                    --folder_path ./knowledge_base \
                    --query "$query" >> answers.txt
  echo "\n---\n" >> answers.txt
done
```

### 🧠 LLM-Only Mode (No Document Retrieval)

**Creative Writing Assistant**
```bash
python docuchat.py --model_path models/llama-2-7b-chat.Q4_K_M.gguf \
                  --no-rag \
                  --system_prompt "You are a creative writing assistant who helps craft engaging stories."
```

**Code Assistant**
```bash
python docuchat.py --model_path models/codellama-7b.Q4_K_M.gguf \
                  --no-rag \
                  --system_prompt "You are an expert programmer who provides clean, efficient code examples and explanations."
```

**Quick Facts and Calculations**
```bash
python docuchat.py --model_path models/llama-2-7b-chat.Q4_K_M.gguf \
                  --no-rag \
                  --query "What is the square root of 144 divided by 3?"
```

### 🔄 Document Management

**Refresh Changed Document Embeddings**
```bash
python docuchat.py --model_path models/llama-2-7b-chat.Q4_K_M.gguf \
                  --folder_path ./my_documents \
                  --refresh
```

**Force Rebuild All Embeddings**
```bash
python docuchat.py --model_path models/llama-2-7b-chat.Q4_K_M.gguf \
                  --folder_path ./my_documents \
                  --force-refresh
```

**Process Documents with Custom Chunking**
```bash
python docuchat.py --model_path models/llama-2-7b-chat.Q4_K_M.gguf \
                  --folder_path ./large_documents \
                  --chunk_size 1500 \
                  --chunk_overlap 300 \
                  --force-refresh
```

### 🔍 Specialized Use Cases

**Technical Documentation Analysis**
```bash
python docuchat.py --model_path models/llama-2-13b-chat.Q5_K_M.gguf \
                  --folder_path ./api_documentation \
                  --n_retrieve 10 \
                  --system_prompt "You are a technical documentation expert. Provide detailed and accurate information about APIs, code examples, and implementation details."
```

**Research Paper Assistant**
```bash
python docuchat.py --model_path models/llama-2-13b-chat.Q5_K_M.gguf \
                  --folder_path ./research_papers \
                  --embedding_model all-mpnet-base-v2 \
                  --system_prompt "You are a research assistant who helps analyze academic papers. Provide detailed summaries, extract key findings, and explain complex concepts in clear language."
```

**Legal Document Analysis**
```bash
python docuchat.py --model_path models/llama-2-13b-chat.Q5_K_M.gguf \
                  --folder_path ./legal_documents \
                  --n_retrieve 8 \
                  --system_prompt "You are a legal assistant who helps analyze contracts, agreements, and legal documents. Identify important clauses, potential issues, and provide clear explanations of legal terminology."
```

### 🛠️ Advanced Configuration

**Using a Different Embedding Model**
```bash
python docuchat.py --model_path models/llama-2-7b-chat.Q4_K_M.gguf \
                  --folder_path ./my_documents \
                  --embedding_model multi-qa-MiniLM-L6-cos-v1 \
                  --force-refresh
```

**Optimizing for Performance**
```bash
python docuchat.py --model_path models/llama-2-7b-chat.Q4_K_M.gguf \
                  --folder_path ./my_documents \
                  --n_ctx 2048 \
                  --max_tokens 1024
```

**Debugging and Troubleshooting**
```bash
python docuchat.py --model_path models/llama-2-7b-chat.Q4_K_M.gguf \
                  --folder_path ./my_documents \
                  --verbose \
                  --log_file debug.log
```

### 📊 Interactive Commands

Once in interactive mode, you can use these special commands:

- `/help` - Show available commands
- `/exit` or `/quit` - Exit the application
- `/clear` - Clear the conversation history
- `/refresh` - Refresh document embeddings
- `/info` - Show system information
- `/rag on|off` - Enable or disable RAG mode
- `/sources on|off` - Show or hide source documents



## 🛡️ Robustness & Error Handling

DocuChat includes comprehensive error handling and robustness features:

### Process Management
- **Automatic Cleanup**: Orphaned processes are automatically terminated
- **Timeout Protection**: Configurable timeouts prevent hanging operations
- **Resource Management**: Memory and file handle cleanup

### Error Recovery
- **Graceful Degradation**: Falls back to core functionality when optional features fail
- **Retry Logic**: Automatic retries for transient failures
- **Detailed Logging**: Comprehensive error reporting with stack traces

### Configuration Validation
- **Schema Validation**: YAML and JSON configuration validation
- **Path Verification**: Automatic path resolution and validation
- **Dependency Checking**: Runtime dependency verification

### Security Features
- **Input Sanitization**: Safe handling of user inputs
- **Path Traversal Protection**: Secure file access controls
- **Process Isolation**: Sandboxed execution for external tools

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

DocuChat implements a sophisticated RAG (Retrieval-Augmented Generation) architecture that combines the power of semantic search with local language models. Here's a detailed breakdown:

### 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Documents     │───▶│  Document        │───▶│  Text Chunks    │
│  (PDF, DOCX,    │    │  Processor       │    │  (Overlapping)  │
│   TXT, MD)      │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  Embedding       │───▶│  Vector         │
│                 │    │  Model           │    │  Embeddings     │
└─────────────────┘    │ (Sentence Trans) │    │                 │
                       └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Final Response │◀───│  Language Model  │◀───│  ChromaDB       │
│  (Streaming)    │    │  (GGUF/llama.cpp)│    │  Vector Store   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### 📊 RAG Mode (Default) - Step by Step

#### Phase 1: Document Ingestion & Processing
1. **📄 Document Loading**: 
   - Scans specified folder recursively for supported files
   - Uses specialized parsers: PyPDF2 (PDF), python-docx (Word), built-in (TXT/MD)
   - Handles encoding detection and error recovery

2. **✂️ Intelligent Chunking**:
   - Splits documents into overlapping chunks (default: 1000 chars with 200 char overlap)
   - Preserves sentence boundaries and paragraph structure
   - Maintains document metadata (filename, chunk index, source location)

3. **🧮 Vector Embedding Generation**:
   - Converts each text chunk into high-dimensional vectors (384 or 768 dimensions)
   - Uses sentence-transformers models (default: all-MiniLM-L6-v2)
   - Captures semantic meaning, not just keyword matching

4. **💾 Persistent Storage**:
   - Stores embeddings in ChromaDB with metadata
   - Creates efficient indexes for fast similarity search
   - Persists between sessions to avoid reprocessing

#### Phase 2: Query Processing & Response Generation
5. **❓ Query Analysis**:
   - User query is embedded using the same model as documents
   - Query vector is compared against all document vectors
   - Uses cosine similarity for semantic matching

6. **🔍 Context Retrieval**:
   - Retrieves top-k most similar document chunks (default: 5)
   - Ranks by relevance score and diversity
   - Includes source attribution for transparency

7. **🤖 LLM Integration**:
   - Constructs prompt with system instructions, retrieved context, and user query
   - Sends to local GGUF model via llama-cpp-python
   - Uses appropriate chat template for the specific model

8. **⚡ Streaming Response**:
   - Generates response token-by-token in real-time
   - Displays source documents used for each response
   - Maintains conversation context within session

### 🚀 LLM-Only Mode (--no-rag)

When RAG is disabled:
- Bypasses document retrieval entirely
- Sends queries directly to the language model
- Useful for general questions or creative tasks
- Faster response times but no document-specific knowledge

### 🔄 Streaming Implementation Details

**Real-time Token Generation**:
- Uses llama-cpp-python's streaming API
- Displays tokens as they're generated (like ChatGPT)
- Provides immediate feedback without waiting for complete response
- Allows interruption of long responses

**Technical Benefits**:
- **Lower Perceived Latency**: Users see progress immediately
- **Better UX**: More engaging and responsive interface
- **Memory Efficient**: Processes tokens incrementally
- **Interruptible**: Can stop generation mid-stream

### 🗄️ Vector Database Architecture

**ChromaDB Implementation**:
- **Storage Location**: `./vectordbs/chroma.sqlite3`
- **Persistence**: Embeddings survive application restarts
- **Efficiency**: Optimized for similarity search operations
- **Scalability**: Handles thousands of documents efficiently

**Metadata Management**:
- **Document Tracking**: Maps chunks back to source files
- **Change Detection**: SHA-256 hashing detects file modifications
- **Incremental Updates**: Only processes new or changed files
- **JSON Serialization**: Secure, human-readable metadata storage

### 🧠 Embedding Model Strategy

**Model Selection Criteria**:
- **Speed vs Quality**: Balance between inference time and accuracy
- **Dimensionality**: Higher dimensions = better quality, more memory
- **Language Support**: Multilingual models for non-English content
- **Domain Specialization**: Q&A optimized models for better retrieval

**Supported Models**:
- `all-MiniLM-L6-v2`: Fast, good quality (384 dims)
- `all-mpnet-base-v2`: High quality, slower (768 dims)
- `multi-qa-MiniLM-L6-cos-v1`: Optimized for Q&A tasks
- Custom models: Any sentence-transformers compatible model

### 🔒 Security & Privacy Architecture

**Local-First Design**:
- **No External APIs**: Everything runs on your machine
- **Data Privacy**: Documents never leave your computer
- **Offline Capable**: Works without internet connection
- **Secure Storage**: SHA-256 hashing, JSON serialization

**Security Measures**:
- **Input Sanitization**: Safe handling of file paths and user input
- **Process Isolation**: Controlled execution environment
- **Memory Management**: Automatic cleanup and resource management
- **Error Boundaries**: Graceful handling of malformed documents

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

This comprehensive troubleshooting guide will help you resolve common issues and optimize DocuChat's performance.

### 🚨 Common Issues & Solutions

#### Model Loading Problems

**❌ Model file not found**
```
Error: Model file not found: models/llama-2-7b-chat.Q4_K_M.gguf
```
**✅ Solutions:**
- Verify the model path is correct and the file exists
- Check file permissions (ensure read access)
- Ensure the model is in GGUF format (not GGML or other formats)
- Download the model if missing:
  ```bash
  # Example download command
  wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf
  ```

**❌ Model loading fails with memory error**
```
Error: Failed to load model: not enough memory
```
**✅ Solutions:**
- Use a smaller quantized model (Q4_K_M instead of Q5_K_M or Q8_0)
- Reduce context window size: `--n_ctx 2048`
- Close other memory-intensive applications
- Consider using a smaller model variant (7B instead of 13B)

**❌ Unsupported model format**
```
Error: Unsupported model format or corrupted file
```
**✅ Solutions:**
- Ensure the model is in GGUF format (newer format)
- Re-download the model if corrupted
- Check file integrity with checksums if available

#### Document Processing Issues

**❌ Document parsing errors**
```
Error: Failed to parse document: example.pdf
```
**✅ Solutions:**
- Ensure documents are not password-protected
- Check if the document is corrupted
- For PDFs: Try converting to text first
- For Word docs: Ensure they're in .docx format (not .doc)
- Use `--verbose` to see detailed error messages

**❌ No documents found**
```
Warning: No supported documents found in folder
```
**✅ Solutions:**
- Check the folder path is correct
- Ensure documents are in supported formats (PDF, DOCX, TXT, MD)
- Verify file extensions are correct
- Check folder permissions

**❌ Embedding generation fails**
```
Error: Failed to generate embeddings for documents
```
**✅ Solutions:**
- Check internet connection (for first-time model download)
- Ensure sufficient disk space for embedding model
- Try a different embedding model: `--embedding_model all-MiniLM-L6-v2`
- Clear the embeddings cache and retry

#### Performance Issues

**❌ Very slow response generation**
**✅ Optimization strategies:**
- **Use GPU acceleration** (if available):
  ```bash
  pip install llama-cpp-python[cuda]  # For NVIDIA GPUs
  ```
- **Reduce model size**: Use Q4_K_M instead of larger quantizations
- **Optimize context window**: `--n_ctx 2048` for faster processing
- **Reduce retrieved chunks**: `--n_retrieve 3` instead of default 5
- **Use faster embedding model**: `--embedding_model all-MiniLM-L6-v2`

**❌ High memory usage**
**✅ Memory optimization:**
- Use smaller chunk sizes: `--chunk_size 500`
- Reduce context window: `--n_ctx 2048`
- Process documents in smaller batches
- Close other applications

#### Configuration Issues

**❌ Configuration file not found**
```
Error: Configuration file not found: config/config.yaml
```
**✅ Solutions:**
- Create the config directory: `mkdir config`
- Copy the example config file
- Use absolute paths in configuration
- Specify config file explicitly: `--config /path/to/config.yaml`

**❌ Invalid configuration values**
```
Error: Invalid temperature value: 2.5
```
**✅ Solutions:**
- Check value ranges (temperature: 0.0-1.0, top_p: 0.0-1.0)
- Ensure numeric values are properly formatted
- Use quotes for string values in YAML
- Validate YAML syntax with online tools

### 🐛 Debug Mode & Logging

#### Enable Comprehensive Debugging
```bash
python docuchat.py --verbose --log_file debug.log
```

#### Log Analysis
**Check these log entries for issues:**
- Model loading progress and memory usage
- Document processing statistics
- Embedding generation times
- Query processing details
- Error stack traces

#### Advanced Debugging
```bash
# Maximum verbosity with detailed timing
python docuchat.py --verbose --log_file debug.log --show_timing

# Debug specific components
DOCUCHAT_DEBUG=1 python docuchat.py --verbose
```

### ⚡ Performance Optimization Guide

#### Hardware Optimization

**For CPU-only systems:**
- Use Q4_K_M quantized models
- Set `n_ctx` to 2048 or lower
- Use `all-MiniLM-L6-v2` embedding model
- Limit concurrent processes

**For GPU systems:**
- Install CUDA-enabled llama-cpp-python
- Use larger models (Q5_K_M or Q8_0)
- Increase context window to 4096+
- Use higher-quality embedding models

#### Model Selection Guide

| System RAM | Recommended Model | Context Window | Performance |
|------------|------------------|----------------|-------------|
| 8GB | 7B Q4_K_M | 2048 | Good |
| 16GB | 7B Q5_K_M | 4096 | Better |
| 32GB+ | 13B Q4_K_M | 4096 | Best |

#### Document Processing Optimization

**For large document collections:**
```bash
# Optimize chunking for better retrieval
python docuchat.py --chunk_size 800 --chunk_overlap 150

# Use high-quality embeddings for better accuracy
python docuchat.py --embedding_model all-mpnet-base-v2

# Increase retrieval for comprehensive answers
python docuchat.py --n_retrieve 8
```

**For quick queries:**
```bash
# Optimize for speed
python docuchat.py --chunk_size 500 --n_retrieve 3 --n_ctx 2048
```

### 🔍 System Requirements Check

#### Verify Your Setup
```bash
# Check Python version (3.8+ required)
python --version

# Check available memory
python -c "import psutil; print(f'Available RAM: {psutil.virtual_memory().available / 1024**3:.1f} GB')"

# Check disk space
df -h .

# Test model loading
python -c "from llama_cpp import Llama; print('llama-cpp-python working')"
```

#### Environment Validation
```bash
# Validate all dependencies
pip check

# Test embedding model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Check ChromaDB
python -c "import chromadb; print('ChromaDB working')"
```

### 📞 Getting Help

If you're still experiencing issues:

1. **Check the logs** with `--verbose --log_file debug.log`
2. **Verify system requirements** and dependencies
3. **Try with minimal configuration** to isolate the issue
4. **Search existing issues** in the project repository
5. **Create a detailed bug report** with:
   - System specifications (OS, RAM, Python version)
   - Complete error messages and logs
   - Steps to reproduce the issue
   - Configuration files used


## 🛠️ Development

### 🏗️ Architecture Overview

DocuChat is built with a clean, modular architecture that separates concerns and enables easy maintenance and extension.

```
┌─────────────────────────────────────────────────────────────┐
│                    DocuChat Application                     │
├─────────────────────────────────────────────────────────────┤
│  CLI Interface  │  Configuration  │  Interactive Shell     │
├─────────────────────────────────────────────────────────────┤
│                    Core Components                          │
├─────────────────┬─────────────────┬─────────────────────────┤
│   DocuChat      │   VectorStore   │   Document Processors   │
│   (Main Logic)  │   (Embeddings)  │   (PDF, DOCX, TXT, MD)  │
├─────────────────┼─────────────────┼─────────────────────────┤
│                    External Dependencies                    │
├─────────────────┬─────────────────┬─────────────────────────┤
│  llama-cpp-     │   ChromaDB      │   sentence-transformers │
│  python         │   (Vector DB)   │   (Embeddings)          │
└─────────────────┴─────────────────┴─────────────────────────┘
```

### 📦 Core Components

#### 1. DocuChatConfig Class
**Purpose**: Centralized configuration management
**Responsibilities**:
- YAML configuration file parsing
- Command-line argument processing
- Configuration validation and defaults
- Environment variable support

**Key Methods**:
- `from_yaml()`: Load configuration from YAML file
- `update_from_args()`: Override with CLI arguments
- `validate()`: Ensure configuration consistency

#### 2. VectorStore Class
**Purpose**: Document embedding and similarity search
**Responsibilities**:
- Document chunking and preprocessing
- Vector embedding generation
- ChromaDB integration and persistence
- Similarity search and retrieval

**Key Methods**:
- `add_documents()`: Process and store document embeddings
- `search()`: Retrieve similar document chunks
- `refresh()`: Update embeddings for changed documents

#### 3. DocuChat Class
**Purpose**: Main application logic and LLM integration
**Responsibilities**:
- Language model loading and management
- RAG pipeline orchestration
- Response generation and streaming
- Interactive chat session management

**Key Methods**:
- `chat()`: Process user queries with RAG
- `generate_response()`: LLM response generation
- `interactive_chat()`: Handle chat sessions

#### 4. Document Processors
**Purpose**: Format-specific document parsing
**Supported Formats**:
- **PDF**: PyPDF2-based extraction with error recovery
- **DOCX**: python-docx for Word document parsing
- **TXT/MD**: Native Python text processing with encoding detection

### 🔧 Dependencies & Technologies

#### Core Dependencies
```python
# LLM Integration
llama-cpp-python>=0.2.0    # GGUF model loading and inference

# Embeddings & Vector Search
sentence-transformers>=2.2.0  # Text embedding generation
chromadb>=0.4.0               # Vector database

# Document Processing
PyPDF2>=3.0.0                 # PDF parsing
python-docx>=0.8.11           # Word document parsing

# Configuration & Utilities
PyYAML>=6.0                   # YAML configuration
colorama>=0.4.6               # Cross-platform colors
psutil>=5.9.0                 # System monitoring
```

#### Optional Dependencies
```python
# GPU Acceleration (optional)
llama-cpp-python[cuda]        # NVIDIA GPU support
llama-cpp-python[metal]       # Apple Metal support

# Advanced Embeddings (optional)
torch>=1.9.0                  # PyTorch for advanced models
transformers>=4.20.0          # Hugging Face transformers
```

### 🚀 Development Setup

#### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-username/docuchat.git
cd docuchat

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt
```

#### 2. Development Dependencies
```bash
# Code quality tools
pip install black isort flake8 mypy

# Testing framework
pip install pytest pytest-cov pytest-mock

# Documentation
pip install sphinx sphinx-rtd-theme
```

#### 3. Pre-commit Hooks
```bash
# Install pre-commit
pip install pre-commit

# Setup hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

### 🧪 Testing

#### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=docuchat --cov-report=html

# Run specific test file
pytest tests/test_vector_store.py

# Run with verbose output
pytest -v
```

#### Test Structure
```
tests/
├── unit/
│   ├── test_config.py
│   ├── test_vector_store.py
│   └── test_document_processors.py
├── integration/
│   ├── test_rag_pipeline.py
│   └── test_cli.py
└── fixtures/
    ├── sample_documents/
    └── test_configs/
```

### 📝 Code Style & Standards

#### Code Formatting
```bash
# Format code with Black
black docuchat.py

# Sort imports with isort
isort docuchat.py

# Check style with flake8
flake8 docuchat.py

# Type checking with mypy
mypy docuchat.py
```

#### Coding Standards
- **PEP 8**: Follow Python style guidelines
- **Type Hints**: Use type annotations for all functions
- **Docstrings**: Google-style docstrings for all public methods
- **Error Handling**: Comprehensive exception handling
- **Logging**: Structured logging with appropriate levels

### 🔄 Contributing Guidelines

#### 1. Development Workflow
```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and commit
git add .
git commit -m "feat: add your feature description"

# Push and create pull request
git push origin feature/your-feature-name
```

#### 2. Commit Message Format
Follow conventional commits:
```
feat: add new feature
fix: bug fix
docs: documentation changes
style: formatting changes
refactor: code refactoring
test: add or update tests
chore: maintenance tasks
```

#### 3. Pull Request Process
1. **Fork** the repository
2. **Create** a feature branch
3. **Write** tests for new functionality
4. **Ensure** all tests pass
5. **Update** documentation if needed
6. **Submit** pull request with clear description

### 🏗️ Extension Points

#### Adding New Document Formats
```python
def process_new_format(file_path: str) -> str:
    """Process a new document format.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        Extracted text content
    """
    # Implementation here
    pass

# Register in document processor
DOCUMENT_PROCESSORS['.new_ext'] = process_new_format
```

#### Custom Embedding Models
```python
class CustomEmbeddingModel:
    def encode(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for input texts."""
        # Custom implementation
        pass
```

#### Plugin System (Future)
```python
# Plugin interface for future extensibility
class DocuChatPlugin:
    def pre_process(self, query: str) -> str:
        """Pre-process user query."""
        return query
    
    def post_process(self, response: str) -> str:
        """Post-process generated response."""
        return response
```

### 📊 Performance Monitoring

#### Profiling
```bash
# Profile application performance
python -m cProfile -o profile.stats docuchat.py --query "test"

# Analyze profile results
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"
```

#### Memory Monitoring
```bash
# Monitor memory usage
python -m memory_profiler docuchat.py

# Line-by-line memory profiling
@profile
def your_function():
    # Function implementation
    pass
```

### 🔒 Security Considerations

- **Input Sanitization**: All user inputs are validated
- **Path Traversal**: File paths are validated and restricted
- **Memory Safety**: Proper resource cleanup and limits
- **Dependency Security**: Regular security audits with `pip-audit`
- **Data Privacy**: All processing happens locally

## 🗺️ Roadmap

DocuChat is under active development with a clear roadmap for future enhancements. Here's what's planned for upcoming releases:

### 🚀 Near-Term (0-3 Months)

#### Performance Enhancements
- **✅ GPU Acceleration Improvements**: Enhanced CUDA support for NVIDIA GPUs
- **✅ Memory Optimization**: Reduced RAM usage for large document collections
- **✅ Batch Processing**: Parallel document processing for faster indexing
- **✅ Caching System**: Smart caching for frequently accessed documents

#### User Experience
- **✅ Command History**: Save and recall previous queries
- **✅ Export Functionality**: Save conversations to markdown or text files
- **✅ Progress Indicators**: Better feedback during long operations
- **✅ Enhanced Source Attribution**: Improved document source display

#### Document Processing
- **✅ HTML Support**: Native processing of HTML documents
- **✅ Code File Support**: Better handling of source code files
- **✅ Table Extraction**: Improved handling of tables in documents
- **✅ Metadata Extraction**: Author, date, title extraction from documents

### 🌟 Mid-Term (3-6 Months)

#### Advanced RAG Features
- **🔄 Hybrid Search**: Combine semantic and keyword search
- **🔄 Re-ranking**: Two-stage retrieval with result re-ranking
- **🔄 Query Decomposition**: Break complex queries into sub-queries
- **🔄 Context Compression**: Optimize retrieved context for relevance

#### UI Improvements
- **🔄 Web Interface**: Simple web UI for easier interaction
- **🔄 Rich Text Formatting**: Markdown and code syntax highlighting
- **🔄 Visualization Tools**: Display document relationships graphically
- **🔄 Mobile-Friendly Design**: Responsive interface for all devices

#### Integration & Extensibility
- **🔄 Plugin System**: Framework for custom extensions
- **🔄 API Mode**: REST API for programmatic access
- **🔄 Integration Hooks**: Connect with other tools and workflows
- **🔄 Custom Pipelines**: Configurable processing pipelines

### 🔮 Long-Term (6+ Months)

#### Multi-Modal Capabilities
- **📅 Image Understanding**: Process and reference images in documents
- **📅 Chart & Graph Analysis**: Extract information from visual data
- **📅 PDF Form Recognition**: Handle structured form documents
- **📅 Handwriting Recognition**: Process handwritten notes

#### Advanced Intelligence
- **📅 Multilingual Support**: Enhanced capabilities for non-English documents
- **📅 Fine-Tuning Tools**: Customize models for specific domains
- **📅 Agent Framework**: Multi-step reasoning and tool use
- **📅 Knowledge Graph**: Build connections between document entities

#### Enterprise Features
- **📅 Multi-User Support**: Shared document collections with permissions
- **📅 Compliance Tools**: Audit logs and access controls
- **📅 Scalability Enhancements**: Distributed processing for large deployments
- **📅 Enterprise Authentication**: LDAP, SAML, and SSO integration

### 🤝 Community Contributions

We welcome contributions in these areas:

- **Documentation Improvements**: Help expand and clarify documentation
- **Bug Fixes**: Address issues in the issue tracker
- **Test Coverage**: Expand unit and integration tests
- **New Document Formats**: Add support for additional file types
- **Performance Optimizations**: Improve speed and resource usage
- **UI Enhancements**: Improve the user experience

## 📄 License

This project is licensed under the MIT License.

## 🤝 Support

- **Issues**: Report bugs and request features on GitHub Issues
- **Documentation**: Check this README for comprehensive guides
- **Community**: Join discussions for help and feature requests

## 🙏 Acknowledgments

DocuChat stands on the shoulders of these amazing open-source projects:

- **[llama-cpp-python](https://github.com/abetlen/llama-cpp-python)**: Efficient C++ implementation of LLM inference with Python bindings
- **[sentence-transformers](https://github.com/UKPLab/sentence-transformers)**: State-of-the-art text embedding models
- **[ChromaDB](https://github.com/chroma-core/chroma)**: Fast, scalable vector database for similarity search
- **[TheBloke](https://huggingface.co/TheBloke)**: Quantized GGUF models that make local LLMs accessible
- **[PyPDF2](https://github.com/py-pdf/PyPDF2)**: PDF document processing library
- **[python-docx](https://github.com/python-openxml/python-docx)**: Word document processing

### Special Thanks

- To all contributors who have helped improve DocuChat
- The open-source AI community for making powerful language models accessible
- Users who provide valuable feedback and bug reports

---

**Happy chatting with your documents! 🚀📚**
