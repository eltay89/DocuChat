# DocuChat v2.0 🤖📚

> **Privacy-focused, local AI document chat assistant with advanced RAG capabilities**

DocuChat v2.0 is a sophisticated, privacy-first RAG (Retrieval-Augmented Generation) application that enables intelligent conversations with your documents using local language models. Built with Python and powered by cutting-edge AI technologies, DocuChat transforms your document collections into an interactive knowledge base.

## 🎯 What is DocuChat?

DocuChat is a local AI-powered document chat system that:

- **Processes documents** (PDF, DOCX, TXT, Markdown, HTML, PPTX, CSV, JSON, Images) into searchable knowledge
- **Uses vector embeddings** to understand document content semantically
- **Leverages local LLMs** for intelligent responses with no external API calls
- **Maintains complete privacy** - all processing happens on your machine
- **Provides real-time streaming** responses for immediate feedback
- **Remembers context** across conversations within a session
- **Supports advanced features** like OCR, hybrid search, and reranking

## 🏗️ Advanced Tool System Architecture

### How Tools Work

#### 1. Tool Discovery and Loading
```python
# Automatic tool discovery from tools/ directory
def discover_tools(config: dict = None, silent: bool = False) -> Dict[str, BaseTool]:
    tools = {}
    tools_dir = os.path.dirname(__file__)
    
    # Scan for Python files and load tool classes
    for filename in os.listdir(tools_dir):
        if filename.endswith('.py') and filename not in ['__init__.py', 'base_tool.py']:
            # Import module and find BaseTool subclasses
            module = importlib.import_module(f'.{module_name}', package='tools')
            # Instantiate tools automatically
```

#### 2. Tool Base Architecture
All tools inherit from the `BaseTool` abstract base class:

```python
class BaseTool(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool identifier for LLM function calling"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description for LLM"""
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """OpenRouter-compatible JSON schema"""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Execute tool with validated parameters"""
        pass
```

#### 3. LLM Integration Process
1. **Schema Generation**: Tools automatically generate OpenRouter-compatible function schemas
2. **LLM Decision Making**: The language model decides which tools to use based on user queries
3. **Parameter Validation**: Tool parameters are validated against JSON schemas
4. **Execution**: Tools execute with proper error handling and logging
5. **Result Integration**: Tool outputs are integrated into the conversation context

### Built-in Tools

#### 🔍 Search Tool
- **Purpose**: Web search using DuckDuckGo
- **Capabilities**: General information retrieval, date queries, current events
- **Security**: Rate limiting, content filtering, safe browsing
- **Return Format**: Structured results with titles, snippets, and URLs

#### 🧮 Calculator Tool
- **Purpose**: Mathematical calculations and expressions
- **Capabilities**: Basic arithmetic, scientific functions, safe evaluation
- **Security**: Sandboxed execution, operator whitelisting
- **Return Format**: Numerical results with step-by-step breakdown

#### 📖 Read File Tool
- **Purpose**: Read and analyze local files
- **Capabilities**: Text files, code files, configuration files
- **Security**: Path validation, access control, size limits
- **Return Format**: File content with metadata and encoding info

#### ✍️ Write File Tool
- **Purpose**: Create and modify files safely
- **Capabilities**: Text creation, code generation, data export
- **Security**: Directory restrictions, backup creation, validation
- **Return Format**: Success confirmation with file details

#### ✅ Task Completion Tool
- **Purpose**: Mark tasks as complete and provide summaries
- **Capabilities**: Status tracking, progress reporting, completion validation
- **Security**: State validation, audit logging
- **Return Format**: Completion status with summary and next steps

### Tool System Benefits

1. **Extensibility**: Easy to add new tools by inheriting from `BaseTool`
2. **Type Safety**: JSON Schema validation for parameters
3. **Error Handling**: Graceful degradation and error reporting
4. **Security**: Built-in path validation and operation restrictions
5. **Performance**: Efficient tool discovery and caching
6. **Debugging**: Comprehensive logging and status reporting

### Creating Custom Tools

```python
from tools.base_tool import BaseTool

class CustomTool(BaseTool):
    def __init__(self, config: dict):
        self.config = config
    
    @property
    def name(self) -> str:
        return "custom_action"
    
    @property
    def description(self) -> str:
        return "Performs a custom action with given parameters"
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "Input parameter"},
                "mode": {"type": "string", "enum": ["fast", "accurate"]}
            },
            "required": ["input"]
        }
    
    def execute(self, **kwargs) -> dict:
        # Tool implementation
        return {"result": "success", "data": processed_data}
```

## ✨ Key Features

### 🚀 Advanced AI Capabilities
- **Real-time Streaming**: Character-by-character response generation
- **Auto-Detecting Chat Templates**: Automatic model format detection (ChatML, Llama2, Alpaca)
- **Local LLM Support**: GGUF format models with no internet required
- **Multi-format Documents**: PDF, DOCX, TXT, MD, HTML, PPTX, CSV, JSON, Images
- **Semantic Vector Search**: ChromaDB with sentence transformers
- **Interactive Chat Interface**: Natural conversation with context awareness
- **Flexible Configuration**: YAML files and command-line arguments
- **Smart Model Integration**: Automatic optimization and template detection
- **Robust Error Handling**: Comprehensive error recovery and timeout protection
- **Source Attribution**: Document source tracking for transparency

### 🔧 Technical Excellence
- **Persistent Vector Database**: ChromaDB for fast startup and efficient search
- **Modern Architecture**: Built on llama-cpp-python's chat completion API
- **Performance Optimized**: Intelligent chunking, caching, and memory management
- **Privacy-First Design**: Zero external API calls
- **Smart Caching System**: Persistent embeddings and metadata
- **Security Focused**: SHA-256 hashing, JSON serialization, secure file handling
- **Type-Safe Code**: Comprehensive type hints throughout

### 🌟 Enhanced Features (v2.0)
- **OCR Support**: Extract text from images and scanned documents
- **Advanced Document Parsing**: Enhanced extraction with unstructured.io
- **Modern Embedding Models**: BGE-M3, E5-large-v2, Nomic Embed support
- **Hybrid Search**: BM25 sparse + dense vector retrieval
- **Cross-encoder Reranking**: Improved result relevance
- **Query Expansion**: Better query understanding and matching
- **Performance Monitoring**: Comprehensive metrics and logging

### 🛠️ Built-in Tools
- **🔍 Web Search**: DuckDuckGo integration for current information
- **🧮 Calculator**: Safe mathematical expression evaluation
- **📖 File Reader**: Secure local file access and analysis
- **✍️ File Writer**: Controlled file creation and modification
- **✅ Task Manager**: Progress tracking and completion validation

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+**: Modern Python installation (3.9+ recommended)
- **Memory**: At least 8GB RAM (16GB+ recommended for larger models)
- **Storage**: 5-50GB free space (depending on model size)
- **GPU (Optional)**: CUDA-compatible GPU for faster inference
- **Operating System**: Windows, macOS, or Linux

### 📦 Installation

#### Automated Installation
```bash
# Clone the repository
git clone https://github.com/eltay89/DocuChat.git
cd DocuChat

# Run the setup script
./scripts/setup.sh  # Linux/macOS
# or
scripts\setup.bat   # Windows
```

#### Manual Installation
```bash
# Clone and setup environment
git clone https://github.com/eltay89/DocuChat.git
cd DocuChat

# Create virtual environment
python -m venv docuchat-env

# Activate environment
# Windows:
docuchat-env\Scripts\activate
# macOS/Linux:
source docuchat-env/bin/activate

# Install dependencies
pip install -r requirements.txt

# For enhanced features (optional)
pip install -e ".[enhanced]"

# For GPU support (optional)
pip install llama-cpp-python[cuda]
```

### 🤖 Download Models

#### Quick Start Models
```bash
# Llama 3.1 8B (Recommended)
wget -P models/ https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

# Phi-3 Mini (Lightweight)
wget -P models/ https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf
```

#### High-Quality Models
```bash
# Llama 3.1 70B (Best quality, requires 32GB+ RAM)
wget -P models/ https://huggingface.co/bartowski/Meta-Llama-3.1-70B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf
```

### ⚙️ Configuration

1. **Copy configuration template**:
   ```bash
   cp config/config_original.yaml.template config/config.yaml
   ```

2. **Edit configuration** to match your setup:
   ```yaml
   model:
     path: "./models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
     context_length: 4096
   
   documents:
     path: "./documents"
     auto_reload: true
   
   enhanced:
     enabled: true
     ocr_enabled: true
   ```

### 📄 Add Documents

Place your documents in the `documents/` folder:
```
documents/
├── research_papers/
│   ├── paper1.pdf
│   └── paper2.pdf
├── manuals/
│   ├── user_guide.docx
│   └── technical_specs.md
└── data/
    ├── dataset.csv
    └── config.json
```

### 💬 Start Chatting

```bash
# Basic usage
python -m docuchat.cli.main

# With enhanced features
python -m docuchat.cli.main --enhanced

# Custom configuration
python -m docuchat.cli.main --config custom_config.yaml
```

## 🐳 Docker Installation

### Prerequisites
- **Docker**: Version 20.10 or higher
- **Docker Compose**: Version 2.0 or higher
- **System Resources**: 8GB+ RAM, 20GB+ storage

### Quick Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/eltay89/DocuChat.git
   cd DocuChat
   ```

2. **Start with Docker Compose**:
   ```bash
   # Start all services
   docker-compose up -d
   
   # Or start specific service
   docker-compose up docuchat-cli
   ```

3. **Access the application**:
   - **CLI Interface**: `docker-compose exec docuchat bash`
   - **Web API**: http://localhost:8000
   - **Streamlit UI**: http://localhost:8501

### Service Configuration

The `docker-compose.yml` includes three main services:

#### 🖥️ CLI Service (`docuchat`)
- **Purpose**: Interactive terminal interface
- **Usage**: Direct document processing and chat
- **Access**: `docker-compose exec docuchat python -m docuchat.cli.main`

#### 🌐 Web API Service (`docuchat-web`)
- **Purpose**: FastAPI REST API
- **Port**: 8000
- **Endpoints**: `/chat`, `/documents`, `/health`
- **Usage**: Programmatic access and integrations

#### 🎨 Streamlit Service (`docuchat-streamlit`)
- **Purpose**: User-friendly web interface
- **Port**: 8501
- **Features**: File upload, chat interface, document management
- **Usage**: Non-technical users and demonstrations

### Volume Mounting

Persistent data is stored in mounted volumes:

```yaml
volumes:
  - ./documents:/app/documents:rw      # Your documents
  - ./models:/app/models:rw            # LLM models
  - ./vector_store:/app/vector_store:rw # Vector database
  - ./config:/app/config:rw            # Configuration files
  - ./logs:/app/logs:rw                # Application logs
```

### Dockerfile Overview

The `Dockerfile` uses a multi-stage build:

1. **Builder Stage**: Compiles dependencies and creates virtual environment
2. **Production Stage**: Minimal runtime with security best practices
   - Non-root user execution
   - Health checks
   - Optimized layer caching
   - Security-focused base image

### Docker Usage Examples

#### Interactive CLI
```bash
# Start interactive session
docker-compose run --rm docuchat

# Run with custom config
docker-compose run --rm docuchat python -m docuchat.cli.main --config /app/config/custom.yaml

# Enhanced mode
docker-compose run --rm docuchat python -m docuchat.cli.main --enhanced
```

#### Web Interface
```bash
# Start web services
docker-compose up docuchat-web docuchat-streamlit

# Custom ports
docker-compose run -p 9000:8000 docuchat-web
```

#### Development Mode
```bash
# Mount source code for development
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

### Environment Variables

Customize behavior with environment variables:

```bash
# Performance tuning
LLAMA_CPP_LOG_LEVEL=2
ANONYMIZED_TELEMETRY=False
PYTHONUNBUFFERED=1

# Resource limits
OMP_NUM_THREADS=4
MKL_NUM_THREADS=4

# Custom paths
DOCUCHAT_CONFIG_PATH=/app/config/custom.yaml
DOCUCHAT_MODEL_PATH=/app/models/custom-model.gguf
```

### Building Custom Images

```bash
# Build with custom tag
docker build -t docuchat:custom .

# Build with build args
docker build --build-arg PYTHON_VERSION=3.11 -t docuchat:py311 .

# Multi-platform build
docker buildx build --platform linux/amd64,linux/arm64 -t docuchat:multiarch .
```

### Troubleshooting

#### Permission Issues
```bash
# Fix volume permissions
sudo chown -R 1000:1000 ./documents ./models ./vector_store ./config ./logs

# Or use user mapping
docker-compose run --user $(id -u):$(id -g) docuchat
```

#### Resource Limits
```bash
# Check container resources
docker stats docuchat-cli

# Increase memory limit in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 16G
```

#### Volume Mount Issues
```bash
# Verify mounts
docker-compose exec docuchat ls -la /app/

# Check SELinux (if applicable)
sudo setsebool -P container_manage_cgroup on
```

### Production Deployment

For production environments:

1. **Use specific image tags**: Avoid `latest` tag
2. **Set resource limits**: Configure memory and CPU limits
3. **Enable logging**: Configure log drivers and rotation
4. **Security scanning**: Scan images for vulnerabilities
5. **Health monitoring**: Set up monitoring and alerting
6. **Backup strategy**: Regular backup of volumes
7. **Update strategy**: Plan for rolling updates

```yaml
# Production docker-compose.override.yml
version: '3.8'
services:
  docuchat:
    image: docuchat:v2.0.1  # Specific version
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
        reservations:
          memory: 2G
          cpus: '1'
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    restart: unless-stopped
```

## 📖 Basic Usage

### Command Examples
```bash
# Ask about your documents
"What are the main findings in the research papers?"

# Get specific information
"How do I configure the database settings?"

# Request summaries
"Summarize the user manual for new employees"

# Ask for analysis
"What trends do you see in the sales data?"
```

### Command Line Options
```bash
# Custom configuration
python -m docuchat.cli.main --config path/to/config.yaml

# Specify model path
python -m docuchat.cli.main --model-path path/to/model.gguf

# Disable RAG (chat only)
python -m docuchat.cli.main --disable-rag

# Enable enhanced features
python -m docuchat.cli.main --enhanced

# Verbose logging
python -m docuchat.cli.main --verbose

# Single query mode
python -m docuchat.cli.main --query "What is this document about?"
```

## 🛠️ Tool Usage Examples

### Real-World Scenarios

#### Research and Analysis
```
User: "I need to research the latest developments in quantum computing and create a summary document."

Agent Response:
1. 🔍 Uses Search Tool to find recent quantum computing news
2. 📖 Uses Read File Tool to analyze existing research papers
3. ✍️ Uses Write File Tool to create comprehensive summary
4. ✅ Uses Task Completion Tool to mark research complete
```

#### Document Processing
```
User: "Calculate the total revenue from the sales data and update the financial report."

Agent Response:
1. 📖 Uses Read File Tool to access sales data CSV
2. 🧮 Uses Calculator Tool to compute totals and percentages
3. ✍️ Uses Write File Tool to update financial report
4. ✅ Uses Task Completion Tool to confirm completion
```

#### Current Information Retrieval
```
User: "What's the current weather in New York and should I postpone my outdoor meeting?"

Agent Response:
1. 🔍 Uses Search Tool to get current weather data
2. 🧮 Uses Calculator Tool to analyze temperature trends
3. ✅ Uses Task Completion Tool to provide recommendation
```

### Tool Chaining Patterns

#### Sequential Processing
```python
# Example: Research → Analysis → Documentation
tools_used = [
    "search_tool",      # Gather information
    "calculator_tool",  # Analyze data
    "write_file_tool",  # Document results
    "task_done_tool"    # Mark complete
]
```

#### Conditional Branching
```python
# Example: Check file → Read or Create
if file_exists:
    use_tool("read_file_tool")
else:
    use_tool("write_file_tool")
```

#### Error Recovery
```python
# Example: Try search → Fallback to local files
try:
    result = use_tool("search_tool")
except NetworkError:
    result = use_tool("read_file_tool")
```

### Best Practices for Tool Usage

#### Tool Selection Guidelines
1. **Search Tool**: Use for current information, facts, and real-time data
2. **Calculator Tool**: Use for mathematical operations and data analysis
3. **Read File Tool**: Use for accessing local documents and files
4. **Write File Tool**: Use for creating reports, summaries, and documentation
5. **Task Completion Tool**: Use to mark milestones and provide status updates

#### Parameter Optimization
- **Search queries**: Be specific and use relevant keywords
- **File paths**: Use absolute paths when possible
- **Calculations**: Break complex expressions into steps
- **File operations**: Validate paths and permissions

#### Error Handling
- **Network issues**: Implement fallback strategies
- **File access**: Check permissions and existence
- **Calculation errors**: Validate input parameters
- **Tool failures**: Provide alternative approaches

#### Performance Considerations
- **Caching**: Reuse results when appropriate
- **Batching**: Combine related operations
- **Timeouts**: Set reasonable limits for operations
- **Resource usage**: Monitor memory and CPU usage

## 💬 Interactive Commands

While chatting, you can use these commands:

- `/help` - Show available commands and tool information
- `/stats` - Display session statistics and performance metrics
- `/tools` - List available tools and their capabilities
- `/history` - Show conversation history
- `/clear` - Clear conversation context
- `/config` - Show current configuration
- `/quit` or `/exit` - Exit the application

## 🛠️ Built-in Tools

### 🧮 Calculator Tool
**Purpose**: Perform mathematical calculations and data analysis

**Capabilities**:
- Basic arithmetic operations (+, -, *, /, %, **)
- Scientific functions (sin, cos, tan, log, sqrt, etc.)
- Statistical calculations (mean, median, std, etc.)
- Safe expression evaluation with operator whitelisting

**Example Usage**:
```
User: "Calculate the compound interest for $10,000 at 5% for 10 years"
Agent: Uses calculator_tool(expression="10000 * (1 + 0.05)**10")
Result: $16,288.95
```

**Security Features**:
- Sandboxed execution environment
- Operator and function whitelisting
- Input validation and sanitization
- Execution timeout protection

### 📁 File Operations

#### 📖 Read File Tool
**Purpose**: Read and analyze local files securely

**Supported Formats**:
- Text files (.txt, .md, .csv, .json, .xml)
- Code files (.py, .js, .html, .css, .sql)
- Configuration files (.yaml, .ini, .conf)
- Log files and structured data

**Example Usage**:
```
User: "What's in the configuration file?"
Agent: Uses read_file_tool(file_path="./config/config.yaml")
Result: Displays file contents with syntax highlighting
```

**Security Features**:
- Path traversal protection
- File size limits (configurable)
- Access control validation
- Encoding detection and handling

#### ✍️ Write File Tool
**Purpose**: Create and modify files with safety controls

**Capabilities**:
- Create new files and directories
- Append to existing files
- Backup creation before modification
- Template-based file generation

**Example Usage**:
```
User: "Create a summary report of our findings"
Agent: Uses write_file_tool(file_path="./reports/summary.md", content="...")
Result: Creates formatted markdown report
```

**Security Features**:
- Directory restriction enforcement
- Automatic backup creation
- File permission validation
- Content sanitization

### 🔍 Web Search Tool
**Purpose**: Retrieve current information from the internet

**Capabilities**:
- General web search using DuckDuckGo
- News and current events
- Technical documentation lookup
- Fact verification and research

**Example Usage**:
```
User: "What's the latest news about renewable energy?"
Agent: Uses search_tool(query="latest renewable energy news 2024")
Result: Returns recent articles with titles, snippets, and URLs
```

**Features**:
- Rate limiting and respectful crawling
- Content filtering and safety
- Result ranking and relevance
- Source attribution and verification

### ✅ Task Management Tool
**Purpose**: Track progress and mark completion

**Capabilities**:
- Mark tasks as complete
- Provide progress summaries
- Generate completion reports
- Track session accomplishments

**Example Usage**:
```
User: "I've finished analyzing the data, mark this task complete"
Agent: Uses task_done_tool(task="Data analysis", summary="Completed analysis of Q3 sales data")
Result: Task marked complete with summary
```

**Features**:
- Progress tracking and reporting
- Completion validation
- Summary generation
- Session state management

## 🔒 File Security Notes

### Read File Tool Security
- **Path Validation**: Prevents directory traversal attacks
- **Access Control**: Respects file system permissions
- **Size Limits**: Configurable maximum file size (default: 10MB)
- **Type Checking**: Validates file types and encoding

### Write File Tool Security
- **Restricted Directories**: Cannot write to system directories
- **Backup Creation**: Automatically backs up existing files
- **Permission Checks**: Validates write permissions
- **Content Validation**: Sanitizes input content

### Best Practices
1. **Use relative paths** when possible
2. **Validate file permissions** before operations
3. **Monitor file sizes** to prevent resource exhaustion
4. **Regular backups** of important files
5. **Audit file operations** through logging

## 🎯 Best Practices

### Tool Usage Guidelines
1. **Be Specific**: Provide clear, detailed requests for better tool selection
2. **Context Matters**: Include relevant context for more accurate results
3. **Verify Results**: Cross-check important information from multiple sources
4. **Security First**: Be cautious with file operations and external searches
5. **Performance**: Use appropriate tools for the task complexity

### Security Considerations
- **File Access**: Tools respect system permissions and security boundaries
- **Network Requests**: Search tool implements rate limiting and safe browsing
- **Data Privacy**: All processing happens locally, no data sent to external APIs
- **Input Validation**: All tool inputs are validated and sanitized
- **Error Handling**: Graceful failure modes prevent system compromise

## ⚙️ Configuration

### Configuration File Structure

```yaml
# config.yaml
model:
  path: "./models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
  context_length: 4096
  max_tokens: 2048
  temperature: 0.7
  top_p: 0.9
  top_k: 40
  repeat_penalty: 1.1
  n_gpu_layers: 0  # Set to -1 for full GPU offload

documents:
  path: "./documents"
  auto_reload: true
  supported_extensions: [".pdf", ".docx", ".txt", ".md"]
  max_file_size: 50  # MB

embeddings:
  model_name: "BAAI/bge-m3"  # Enhanced mode
  # model_name: "all-MiniLM-L6-v2"  # Basic mode
  chunk_size: 1000
  chunk_overlap: 200
  batch_size: 32

rag:
  enabled: true
  top_k: 5
  similarity_threshold: 0.7
  rerank_enabled: true  # Enhanced mode only
  hybrid_search: true   # Enhanced mode only

chat:
  streaming: true
  max_history: 10
  system_prompt: "You are a helpful AI assistant..."
  temperature: 0.7

ui:
  theme: "dark"
  show_sources: true
  auto_scroll: true
  typing_speed: 50  # Characters per second

logging:
  level: "INFO"
  file: "./logs/docuchat.log"
  max_size: 10  # MB
  backup_count: 5

tools:
  enabled: true
  search_enabled: true
  file_operations_enabled: true
  calculator_enabled: true
  max_search_results: 10
  file_size_limit: 10  # MB

enhanced:
  enabled: false  # Set to true for enhanced features
  ocr_enabled: true
  ocr_languages: ["en", "es", "fr"]
  advanced_parsing: true
  reranking_model: "BAAI/bge-reranker-base"
  query_expansion: true
```

### Environment Variables

```bash
# Model configuration
DOCUCHAT_MODEL_PATH="./models/custom-model.gguf"
DOCUCHAT_CONTEXT_LENGTH=8192
DOCUCHAT_GPU_LAYERS=-1

# Document processing
DOCUCHAT_DOCUMENTS_PATH="./my-documents"
DOCUCHAT_AUTO_RELOAD=true

# Performance tuning
DOCUCHAT_BATCH_SIZE=16
DOCUCHAT_CHUNK_SIZE=512
DOCUCHAT_TOP_K=10

# Enhanced features
DOCUCHAT_ENHANCED=true
DOCUCHAT_OCR_ENABLED=true
DOCUCHAT_HYBRID_SEARCH=true

# Logging
DOCUCHAT_LOG_LEVEL=DEBUG
DOCUCHAT_LOG_FILE="./logs/debug.log"

# Security
DOCUCHAT_TOOLS_ENABLED=true
DOCUCHAT_SEARCH_ENABLED=false
DOCUCHAT_FILE_SIZE_LIMIT=5
```

## 📁 Project Structure

```
DocuChat/
├── src/
│   └── docuchat/
│       ├── __init__.py
│       ├── cli/
│       │   ├── __init__.py
│       │   └── main.py              # Main CLI application
│       ├── core/
│       │   ├── __init__.py
│       │   ├── config.py            # Configuration management
│       │   ├── docuchat.py          # Core chat logic
│       │   ├── document_processor.py # Basic document processing
│       │   ├── enhanced_document_processor.py # Enhanced processing
│       │   ├── vector_store.py      # Basic vector storage
│       │   └── enhanced_vector_store.py # Enhanced vector storage
│       └── utils/
│           ├── __init__.py
│           ├── file_monitor.py      # File system monitoring
│           └── logger.py            # Logging utilities
├── tools/
│   ├── __init__.py                  # Tool discovery
│   ├── base_tool.py                 # Abstract base class
│   ├── calculator_tool.py           # Mathematical calculations
│   ├── read_file_tool.py            # File reading operations
│   ├── search_tool.py               # Web search functionality
│   ├── task_done_tool.py            # Task completion tracking
│   └── write_file_tool.py           # File writing operations
├── config/
│   └── config_original.yaml.template # Configuration template
├── documents/                       # Your documents go here
├── models/                          # LLM models storage
├── vector_store/                    # Vector database storage
├── logs/                           # Application logs
├── tests/                          # Unit and integration tests
├── scripts/                        # Setup and utility scripts
├── requirements.txt                # Python dependencies
├── setup.py                        # Package installation
├── Dockerfile                      # Container definition
├── docker-compose.yml              # Multi-service orchestration
├── .dockerignore                   # Docker build exclusions
├── .gitignore                      # Git exclusions
└── README.md                       # This file
```

## 🧪 Testing

### Running Tests
```bash
# Install test dependencies
pip install -e ".[test]"

# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=docuchat --cov-report=html

# Run specific test categories
python -m pytest tests/unit/
python -m pytest tests/integration/
python -m pytest tests/tools/

# Run performance tests
python -m pytest tests/performance/ --benchmark-only
```

### Test Categories
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow testing
- **Tool Tests**: Tool functionality and security testing
- **Performance Tests**: Benchmarking and optimization
- **Security Tests**: Vulnerability and safety testing

## 🔧 Adding Custom Tools

### Step 1: Create Tool Class
```python
# tools/my_custom_tool.py
from tools.base_tool import BaseTool
import requests

class MyCustomTool(BaseTool):
    @property
    def name(self) -> str:
        return "MyTool"
    
    @property
    def description(self) -> str:
        return "Description of what my tool does"
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "Input parameter"}
            },
            "required": ["input"]
        }
    
    def execute(self, **kwargs) -> dict:
        input_value = kwargs.get("input")
        # Tool implementation here
        result = self.process_input(input_value)
        return {"result": result, "status": "success"}
    
    def process_input(self, input_value: str) -> str:
        # Custom logic implementation
        return f"Processed: {input_value}"
```

### Step 2: Tool Discovery
Tools are automatically discovered when placed in the `tools/` directory. The system will:
1. Scan for Python files in the tools directory
2. Import modules and find BaseTool subclasses
3. Instantiate tools and register them
4. Generate OpenRouter-compatible schemas

### Step 3: Testing Your Tool
```python
# tests/tools/test_my_custom_tool.py
import pytest
from tools.my_custom_tool import MyCustomTool

def test_my_custom_tool():
    tool = MyCustomTool({})
    result = tool.execute(input="test")
    assert result["status"] == "success"
    assert "Processed: test" in result["result"]
```

## 🔧 Troubleshooting

### Common Issues

#### Model Loading Problems
```bash
# Check model file exists and is readable
ls -la models/

# Verify model format (should be .gguf)
file models/your-model.gguf

# Check available memory
free -h

# Try smaller model or reduce context length
python -m docuchat.cli.main --model-path models/smaller-model.gguf
```

#### Document Processing Issues
```bash
# Check document permissions
ls -la documents/

# Verify supported formats
python -c "from docuchat.core.document_processor import DocumentProcessor; print(DocumentProcessor().get_supported_formats())"

# Clear vector store cache
rm -rf vector_store/

# Rebuild embeddings
python -m docuchat.cli.main --rebuild-index
```

#### Performance Issues
```bash
# Monitor resource usage
top -p $(pgrep -f docuchat)

# Check GPU utilization (if using GPU)
nvidia-smi

# Reduce batch size and chunk size
export DOCUCHAT_BATCH_SIZE=8
export DOCUCHAT_CHUNK_SIZE=512

# Enable performance profiling
python -m docuchat.cli.main --profile
```

#### Tool-Related Issues
```bash
# List available tools
python -c "from tools import discover_tools; print(list(discover_tools().keys()))"

# Test specific tool
python -c "from tools.search_tool import SearchTool; tool = SearchTool({}); print(tool.execute(query='test'))"

# Disable problematic tools
export DOCUCHAT_SEARCH_ENABLED=false
```

### Performance Optimization

#### Memory Optimization
- Use smaller models (Q4_K_M instead of Q8_0)
- Reduce context length and chunk size
- Enable model quantization
- Use streaming responses

#### Speed Optimization
- Enable GPU acceleration (if available)
- Increase batch size for embeddings
- Use SSD storage for vector database
- Enable caching for repeated queries

#### Resource Monitoring
```bash
# Monitor memory usage
watch -n 1 'ps aux | grep docuchat'

# Check disk usage
du -sh vector_store/ models/ documents/

# Monitor network usage (for search tool)
netstat -i
```

## 🔒 Privacy and Security

### Privacy Features
- **Local Processing**: All AI processing happens on your machine
- **No External APIs**: No data sent to external services (except optional web search)
- **Secure Storage**: Documents and embeddings stored locally
- **No Telemetry**: No usage data collection or tracking

### Security Measures
- **Input Validation**: All inputs validated and sanitized
- **Path Security**: File operations restricted to safe directories
- **Sandboxed Execution**: Tools run in controlled environments
- **Error Handling**: Secure error messages without information leakage
- **Access Control**: Respects file system permissions

### Best Practices
1. **Keep models updated** to latest versions
2. **Regular security scans** of dependencies
3. **Limit file access** to necessary directories
4. **Monitor tool usage** through logging
5. **Use strong passwords** for any external integrations

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone repository
git clone https://github.com/eltay89/DocuChat.git
cd DocuChat

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
python -m pytest
```

### Contribution Areas
- **New Tools**: Add functionality with custom tools
- **Document Processors**: Support for new file formats
- **UI Improvements**: Enhanced user interface features
- **Performance**: Optimization and efficiency improvements
- **Documentation**: Tutorials, examples, and guides
- **Testing**: Comprehensive test coverage

## 📚 Additional Documentation

- [📖 Installation Guide](INSTALLATION.md) - Detailed setup instructions
- [🏗️ Architecture Overview](ARCHITECTURE.md) - System design and components
- [🔌 API Documentation](API.md) - Programming interfaces
- [🤝 Contributing Guidelines](CONTRIBUTING.md) - How to contribute
- [💬 System Prompts](SYSTEM_PROMPTS.md) - AI behavior configuration
- [📋 Changelog](CHANGELOG.md) - Version history and updates

## 📈 Version History

### v2.0.0 (Current)
- ✨ Enhanced document processing with OCR support
- 🔍 Modern embedding models (BGE-M3, E5-large-v2)
- 🔄 Hybrid search with BM25 + dense retrieval
- 🎯 Cross-encoder reranking for better results
- 🛠️ Comprehensive tool system with automatic discovery
- 🐳 Full Docker containerization support
- 🌐 Multiple interface options (CLI, API, Web UI)
- 🔒 Enhanced security and privacy features
- 📊 Performance monitoring and optimization
- 📚 Comprehensive documentation and examples

### v1.0.1 (Legacy)
- 📄 Basic document processing (PDF, DOCX, TXT, MD)
- 🔍 Simple vector search with ChromaDB
- 💬 Interactive chat interface
- ⚙️ YAML configuration support
- 🔄 Real-time file monitoring
- 🛡️ Basic error handling and logging

---

**DocuChat v2.0** - Transform your documents into an intelligent, conversational knowledge base with the power of local AI. 🚀

For support and questions, please visit our [GitHub Issues](https://github.com/eltay89/DocuChat/issues) page.