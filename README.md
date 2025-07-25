# DocuChat v2.0 🚀

**Advanced AI-Powered Document Chat System with Intelligent Tool Integration**

DocuChat v2.0 is a sophisticated document analysis and chat system that combines the power of local Large Language Models (LLMs) with an intelligent tool ecosystem. Built for privacy, performance, and extensibility, it enables natural conversations with your documents while providing autonomous AI capabilities through a comprehensive tool system.

## 🙏 Acknowledgments

Special thanks to **Pietro Schirano** and **Doriandarko** for their foundational work on AI tools and agent systems. Their innovative approaches to tool integration and LLM orchestration have significantly influenced the design of DocuChat's tool system.

### Pietro Schirano 🎯
**Passionate AI engineer on a mission to democratize AI tools**
- **Founder at MagicPath** 🎨✨ ([magicpath.ai](https://magicpath.ai))
- **EverArt** ([everart.ai](https://everart.ai))
- **Email**: pietro.schirano@gmail.com
- **Twitter**: [@skirano](https://twitter.com/skirano)
- **GitHub**: [@Doriandarko](https://github.com/Doriandarko)
- **Community**: 2.4k followers

Pietro's innovative work on AI tool implementations and agent architectures has been instrumental in shaping DocuChat's intelligent tool system. His vision for democratizing AI tools aligns perfectly with DocuChat's mission to make advanced AI capabilities accessible to everyone.

## 🛠️ Advanced Tool System Architecture

DocuChat v2.0 features a sophisticated tool system that enables the LLM to perform actions beyond text generation. This system is inspired by modern AI agent frameworks and provides a robust foundation for extending the assistant's capabilities.

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
        """Tool name for LLM function calling"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for LLM understanding"""
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """JSON Schema for tool parameters"""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters"""
        pass
```

#### 3. LLM-Tool Integration Process

**Step 1: Schema Generation**
```python
# Convert tools to OpenRouter/OpenAI function calling format
tool_schemas = [tool.to_openrouter_schema() for tool in self.tools]

# Schema format:
{
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search for information using DuckDuckGo",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "description": "Maximum results"}
            },
            "required": ["query"]
        }
    }
}
```

**Step 2: LLM Decision Making**
The LLM receives the tool schemas and decides when to use tools based on:
- User query analysis
- Available tool capabilities
- Context requirements
- Task complexity

**Step 3: Tool Call Execution**
```python
# LLM generates structured tool calls
if 'tool_calls' in message and message['tool_calls']:
    for tool_call in message['tool_calls']:
        tool_name = tool_call['function']['name']
        tool_args = json.loads(tool_call['function']['arguments'])
        
        # Execute tool and capture result
        tool_result = self.tool_map[tool_name].execute(**tool_args)
        
        # Add result back to conversation context
        messages.append({"role": "tool", "name": tool_name, "content": str(tool_result)})
```

**Step 4: Result Integration**
Tool results are seamlessly integrated into the conversation flow, allowing the LLM to:
- Process tool outputs
- Chain multiple tool calls
- Provide contextual responses
- Handle errors gracefully

### Built-in Tools

#### 🔍 Search Tool
- **Purpose**: Web search using DuckDuckGo
- **Capabilities**: General information retrieval, date queries, current events
- **Return Format**: Structured dictionary with summary and sources
- **Special Features**: Smart date detection, content extraction, error handling

#### 🧮 Calculator Tool
- **Purpose**: Mathematical calculations and evaluations
- **Capabilities**: Basic arithmetic, trigonometry, logarithms, constants
- **Security**: Safe AST-based evaluation, restricted operations
- **Return Format**: Expression, result, and success status

#### 📖 Read File Tool
- **Purpose**: Read and analyze local files
- **Capabilities**: Full file reading, head/tail operations, encoding detection
- **Security**: Path validation, restricted directory access
- **Return Format**: File path, content, and metadata

#### ✍️ Write File Tool
- **Purpose**: Create and modify files
- **Capabilities**: Atomic file operations, directory creation, encoding handling
- **Security**: Path sanitization, restricted directory protection
- **Return Format**: Success status, bytes written, file path

#### ✅ Task Completion Tool
- **Purpose**: Signal task completion and exit agent loop
- **Capabilities**: Task summarization, completion messaging
- **Return Format**: Status, summary, timestamp

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
                "input_data": {
                    "type": "string",
                    "description": "Input data for processing"
                }
            },
            "required": ["input_data"]
        }
    
    def execute(self, input_data: str) -> dict:
        try:
            # Custom logic here
            result = self.process_data(input_data)
            return {
                "success": True,
                "result": result,
                "message": "Processing completed successfully"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
```

## 🌟 Key Features

### Core Functionality
- **Multi-format Document Support**: PDF, DOCX, TXT, MD, HTML, PPTX, CSV, JSON
- **Local AI Processing**: Complete privacy with local LLM inference using llama.cpp
- **Advanced RAG System**: Intelligent document retrieval and context-aware responses
- **Real-time File Monitoring**: Automatic document updates and re-indexing
- **Interactive Terminal Interface**: Rich, user-friendly command-line experience

## 🤖 LLM Integration and Agent Behavior

### How the LLM Processes Tool Calls

DocuChat's LLM integration follows a sophisticated agent pattern that enables autonomous tool usage:

#### 1. System Prompt Engineering
```python
# System prompts define tool usage behavior
system_prompt = """
You are an intelligent assistant with access to tools. When you need to:
- Search for information: Use the 'search' tool
- Perform calculations: Use the 'calculate' tool  
- Read files: Use the 'read_file' tool
- Write files: Use the 'write_file' tool
- Complete tasks: Use the 'mark_task_complete' tool

Always use tools when they can help answer the user's question.
"""
```

#### 2. Agent Loop Architecture
```python
def stream_response(self, query: str, context: str = None) -> Generator[str, None, None]:
    messages = self._create_messages(query, context)
    tool_schemas = [tool.to_openrouter_schema() for tool in self.tools]
    
    max_iterations = 5  # Prevent infinite loops
    iteration = 0
    
    while iteration < max_iterations:
        # Generate response with tool schemas
        response = self.generate_response(messages, tools=tool_schemas)
        message = response['choices'][0]['message']
        
        # Check for tool calls
        if 'tool_calls' in message and message['tool_calls']:
            # Execute tools and continue conversation
            for tool_call in message['tool_calls']:
                tool_result = self.execute_tool(tool_call)
                messages.append({"role": "tool", "content": str(tool_result)})
        else:
            # No more tools needed, stream final response
            yield from self.stream_final_response(message)
            break
        
        iteration += 1
```

#### 3. Tool Call Formats

DocuChat supports multiple tool call formats for maximum LLM compatibility:

**Structured Tool Calls (OpenAI Format)**
```json
{
  "tool_calls": [
    {
      "id": "call_123",
      "type": "function",
      "function": {
        "name": "search",
        "arguments": "{\"query\": \"current weather\", \"max_results\": 3}"
      }
    }
  ]
}
```

**Text-based Tool Calls (Fallback)**
```xml
<tool_call>
{
  "name": "search",
  "arguments": {
    "query": "current weather",
    "max_results": 3
  }
}
</tool_call>
```

#### 4. Error Handling and Recovery

```python
try:
    tool_result = self.tool_map[tool_name].execute(**tool_args)
    messages.append({"role": "tool", "name": tool_name, "content": str(tool_result)})
except Exception as e:
    # Graceful error handling
    error_message = f"Tool '{tool_name}' failed: {str(e)}"
    messages.append({"role": "tool", "name": tool_name, "content": error_message})
    # LLM can adapt and try alternative approaches
```

### Agent Capabilities

#### Multi-Step Reasoning
The agent can chain multiple tool calls to solve complex tasks:
1. **Search** for current information
2. **Calculate** based on retrieved data
3. **Write** results to a file
4. **Mark task complete** when finished

#### Context Awareness
- Maintains conversation history across tool calls
- Integrates tool results into response context
- Adapts strategy based on tool success/failure

#### Autonomous Decision Making
The LLM autonomously decides:
- Which tools to use and when
- How to interpret tool results
- Whether to chain additional tool calls
- When a task is complete

### Performance Optimizations

#### Streaming Integration
```python
# Real-time tool execution feedback
yield f"\n\n🔧 Using {tool_name} tool...\n"
tool_result = self.execute_tool(tool_call)
yield f"✅ Tool result: {str(tool_result)[:200]}...\n\n"
```

#### Efficient Tool Loading
```python
# Lazy loading and caching
self.tools, self.tool_map = self._init_tools()
tool_schemas = [tool.to_openrouter_schema() for tool in self.tools]
```

#### Memory Management
- Automatic conversation pruning
- Tool result truncation for large outputs
- Efficient message history management

### Enhanced Features (v2.0)
- **OCR Capabilities**: Extract text from images and scanned documents
- **Advanced Document Parsing**: Powered by unstructured.io for 10+ file formats
- **Modern Embedding Models**: BGE-M3, E5-large-v2, Nomic Embed support
- **Hybrid Search**: BM25 sparse retrieval + dense vector search
- **Cross-encoder Reranking**: Improved result relevance
- **Query Expansion**: Enhanced search capabilities
- **Performance Monitoring**: Comprehensive logging and system metrics

### Built-in Tools
- **Calculator**: Perform mathematical calculations
- **File Operations**: Read and write files
- **Web Search**: Search the internet for current information
- **Task Management**: Mark tasks as complete

## 🚀 Quick Start

### Prerequisites

- **Python 3.8 or higher** (for local installation)
- **Docker & Docker Compose** (for containerized deployment)
- **4GB+ RAM** (8GB+ recommended for larger models)
- **2GB+ free disk space**
- **Optional**: CUDA-compatible GPU for acceleration

## 🐳 Docker Installation (Recommended)

### 1. Clone and Setup
```bash
git clone https://github.com/eltay89/DocuChat.git
cd DocuChat
```

### 2. Quick Start with Docker Compose
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f docuchat-cli

# Access different interfaces:
# CLI: docker exec -it docuchat-cli python docuchat.py
# Web API: http://localhost:8000
# Streamlit: http://localhost:8501
```

### 3. Docker Services Overview

| Service | Port | Description | Access |
|---------|------|-------------|--------|
| **CLI** | - | Interactive terminal interface | `docker exec -it docuchat-cli python docuchat.py` |
| **Web API** | 8000 | FastAPI REST endpoints | http://localhost:8000 |
| **Streamlit** | 8501 | Web-based chat interface | http://localhost:8501 |

### 4. Volume Persistence
```yaml
volumes:
  - ./documents:/app/documents     # Document storage
  - ./models:/app/models          # Model cache
  - ./config:/app/config          # Configuration
  - ./embeddings:/app/embeddings  # Vector embeddings
```

### 5. Environment Configuration
```bash
# Copy and customize configuration
cp config/config.yaml.template config/config.yaml

# Edit configuration for your needs
nano config/config.yaml
```

## 📦 Local Installation

### 1. System Requirements
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.8+ python3-pip git

# macOS
brew install python@3.8 git

# Windows
# Install Python 3.8+ from python.org
# Install Git from git-scm.com
```

### 2. Installation Steps
```bash
# Clone repository
git clone https://github.com/eltay89/DocuChat.git
cd DocuChat

# Create virtual environment
python -m venv docuchat_env

# Activate environment
# Linux/macOS:
source docuchat_env/bin/activate
# Windows:
docuchat_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup configuration
cp config/config.yaml.template config/config.yaml
```

### 3. Configuration Setup
```yaml
# config/config.yaml
llm:
  provider: "openrouter"  # or "ollama", "openai"
  model: "microsoft/wizardlm-2-8x22b"
  api_key: "your-api-key-here"
  base_url: "https://openrouter.ai/api/v1"

rag:
  enabled: true
  chunk_size: 1000
  chunk_overlap: 200
  top_k: 5

embedding:
  model: "BAAI/bge-m3"
  device: "auto"  # "cpu", "cuda", "mps"

enhanced:
  enabled: true  # Enable advanced features
  ocr_enabled: true
  reranking_enabled: true
```

## 🎯 Usage Examples

### CLI Interface
```bash
# Start interactive session
python docuchat.py

# Process documents and chat
> help
> status
> What are the main topics in my documents?
> Calculate the ROI from the financial report
> Search for recent AI developments
```

### Web API Usage
```bash
# Start web server
python -m docuchat.web.api

# API endpoints:
# POST /chat - Send chat messages
# POST /upload - Upload documents
# GET /status - System status
# GET /docs - API documentation
```

### Streamlit Interface
```bash
# Start Streamlit app
streamlit run src/docuchat/web/streamlit_app.py

# Features:
# - File upload interface
# - Real-time chat
# - Document management
# - System monitoring
```

## 🧪 Testing

### Running Tests
```bash
# Run all tests
python run_tests.py

# Run specific test files
python run_tests.py --file test_tools.py

# Run with coverage
python run_tests.py --coverage

# Using pytest directly
pytest tests/ -v
pytest tests/ --cov=src/ --cov-report=html
```

### Test Coverage
The project includes comprehensive tests for:
- **Document Processing**: Text extraction, chunking, metadata handling
- **Vector Store Operations**: Embedding, search, persistence
- **Tool System**: All built-in tools and discovery mechanism
- **API Endpoints**: Web interface and REST API
- **Error Handling**: Edge cases and failure scenarios

### Available Test Files
- `tests/test_document_processor.py` - Document processing functionality
- `tests/test_vector_store.py` - Vector storage and retrieval
- `tests/test_tools.py` - Tool system and individual tools
- `tests/__init__.py` - Test suite initialization

## 📖 SYSTEM_PROMPTS.md

The `SYSTEM_PROMPTS.md` file serves as the core configuration for AI system prompt engineering in DocuChat v2.0:

### Purpose and Functionality
- **AI System Prompt Engineering**: Defines how the AI assistant behaves and responds to user queries
- **Tool Integration**: Enables JSON-based tool usage (Calculator, ReadFile, Search, TaskDone, WriteFile)
- **Response Formatting**: Ensures consistent output with specific guidelines (e.g., `\boxed{answer}` for mathematical results)
- **Real-time Information Access**: Configures the Search tool for current information retrieval
- **Customization**: Allows modification of AI behavior through prompt engineering

### Configuration Location
System prompts are stored in `config/system_prompts/` and can be customized for different use cases:
- `default.txt` - Standard assistant behavior
- `technical.txt` - Technical documentation focus
- `creative.txt` - Creative writing assistance
- `analytical.txt` - Data analysis and research

### Customization
Users can modify system prompts to:
- Change AI personality and tone
- Add domain-specific knowledge
- Configure tool usage patterns
- Set response formatting preferences
- Define conversation flow rules

---

**DocuChat v2.0** - Empowering intelligent document conversations with privacy, performance, and extensibility. 🚀