#!/usr/bin/env python3
"""
DocuChat - A RAG-based document chat application using local LLMs

This application allows you to chat with your documents using local language models.
It supports various document formats and uses vector embeddings for efficient retrieval.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import yaml

try:
    from llama_cpp import Llama
except ImportError:
    print("Error: llama-cpp-python is not installed.")
    print("Please install it with: pip install llama-cpp-python")
    sys.exit(1)

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    print("Error: chromadb is not installed.")
    print("Please install it with: pip install chromadb")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: sentence-transformers is not installed.")
    print("Please install it with: pip install sentence-transformers")
    sys.exit(1)

try:
    import PyPDF2
except ImportError:
    print("Error: PyPDF2 is not installed.")
    print("Please install it with: pip install PyPDF2")
    sys.exit(1)

try:
    from docx import Document as DocxDocument
except ImportError:
    print("Error: python-docx is not installed.")
    print("Please install it with: pip install python-docx")
    sys.exit(1)

try:
    from colorama import Fore, Style, init
    init(autoreset=True)  # Initialize colorama
    COLORS_AVAILABLE = True
except ImportError:
    # Fallback if colorama is not available
    class Fore:
        BLUE = CYAN = GREEN = YELLOW = RED = MAGENTA = WHITE = ""
    class Style:
        BRIGHT = RESET_ALL = ""
    COLORS_AVAILABLE = False


class DocumentProcessor:
    """Handles loading and processing of various document formats."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_documents(self, folder_path: str) -> List[str]:
        """Load and process documents from a folder."""
        documents = []
        folder = Path(folder_path)
        
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        for file_path in folder.rglob('*'):
            if file_path.is_file():
                try:
                    content = self._load_file(file_path)
                    if content:
                        chunks = self._chunk_text(content)
                        documents.extend(chunks)
                        logging.info(f"Loaded {len(chunks)} chunks from {file_path}")
                except Exception as e:
                    logging.warning(f"Failed to load {file_path}: {e}")
        
        return documents
    
    def _load_file(self, file_path: Path) -> Optional[str]:
        """Load content from a single file based on its extension."""
        suffix = file_path.suffix.lower()
        
        try:
            if suffix == '.txt':
                return self._load_text(file_path)
            elif suffix == '.pdf':
                return self._load_pdf(file_path)
            elif suffix in ['.docx', '.doc']:
                return self._load_docx(file_path)
            elif suffix == '.md':
                return self._load_text(file_path)
            else:
                logging.warning(f"Unsupported file type: {suffix}")
                return None
        except Exception as e:
            logging.error(f"Error loading {file_path}: {e}")
            return None
    
    def _load_text(self, file_path: Path) -> str:
        """Load plain text file."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    def _load_pdf(self, file_path: Path) -> str:
        """Load PDF file."""
        text = ""
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def _load_docx(self, file_path: Path) -> str:
        """Load DOCX file."""
        doc = DocxDocument(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings
                for i in range(end, max(start + self.chunk_size // 2, end - 100), -1):
                    if text[i] in '.!?\n':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        return chunks


class VectorStore:
    """Manages vector embeddings and similarity search using ChromaDB."""
    
    def __init__(self, collection_name: str = "documents", embedding_model: str = "all-MiniLM-L6-v2"):
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        
        # Initialize embedding model
        logging.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB
        self.client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            allow_reset=True
        ))
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(collection_name)
            logging.info(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(collection_name)
            logging.info(f"Created new collection: {collection_name}")
    
    def add_documents(self, documents: List[str]):
        """Add documents to the vector store."""
        if not documents:
            logging.warning("No documents to add")
            return
        
        logging.info(f"Generating embeddings for {len(documents)} documents...")
        embeddings = self.embedding_model.encode(documents, show_progress_bar=True)
        
        # Prepare data for ChromaDB
        ids = [f"doc_{i}" for i in range(len(documents))]
        
        # Clear existing collection and add new documents
        self.collection.delete()
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            ids=ids
        )
        
        logging.info(f"Added {len(documents)} documents to vector store")
    
    def search(self, query: str, n_results: int = 5) -> List[str]:
        """Search for similar documents."""
        query_embedding = self.embedding_model.encode([query])
        
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results
        )
        
        return results['documents'][0] if results['documents'] else []


class DocuChatConfig:
    """Configuration class for DocuChat application."""
    
    def __init__(self):
        # Model configuration
        self.model_path = None
        self.n_ctx = 4096
        self.n_threads = None
        self.n_gpu_layers = 0
        self.temperature = 0.7
        self.max_tokens = 2048
        self.top_p = 0.9
        self.top_k = 40
        self.repeat_penalty = 1.1
        
        # Document processing
        self.folder_path = "./documents"  # Default to ./documents folder
        self.chunk_size = 1000
        self.chunk_overlap = 200
        
        # Embeddings and vector store
        self.embedding_model = "all-MiniLM-L6-v2"
        self.collection_name = "documents"
        
        # RAG configuration
        self.n_retrieve = 5
        self.similarity_threshold = 0.7
        self.no_rag = False
        
        # Chat configuration
        self.system_prompt = "You are a helpful assistant that answers questions based on the provided context."
        self.chat_template = "auto"
        
        # UI and interaction
        self.verbose = False
        self.interactive = True
        
        # Performance
        self.batch_size = 512
        self.use_mlock = False
        self.use_mmap = True
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'DocuChatConfig':
        """Load configuration from YAML file."""
        config = cls()
        
        if not os.path.exists(yaml_path):
            logging.warning(f"Config file not found: {yaml_path}. Using defaults.")
            return config
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
            
            # Map YAML structure to config attributes
            if 'model' in yaml_config:
                model_config = yaml_config['model']
                config.model_path = model_config.get('path')
                config.n_ctx = model_config.get('context_length', config.n_ctx)
                config.n_threads = model_config.get('threads')
                config.n_gpu_layers = model_config.get('gpu_layers', config.n_gpu_layers)
                config.temperature = model_config.get('temperature', config.temperature)
                config.max_tokens = model_config.get('max_tokens', config.max_tokens)
                config.top_p = model_config.get('top_p', config.top_p)
                config.top_k = model_config.get('top_k', config.top_k)
                config.repeat_penalty = model_config.get('repeat_penalty', config.repeat_penalty)
            
            if 'documents' in yaml_config:
                doc_config = yaml_config['documents']
                config.folder_path = doc_config.get('folder_path')
                config.chunk_size = doc_config.get('chunk_size', config.chunk_size)
                config.chunk_overlap = doc_config.get('chunk_overlap', config.chunk_overlap)
            
            if 'embeddings' in yaml_config:
                emb_config = yaml_config['embeddings']
                config.embedding_model = emb_config.get('model', config.embedding_model)
            
            if 'vector_store' in yaml_config:
                vs_config = yaml_config['vector_store']
                config.collection_name = vs_config.get('collection_name', config.collection_name)
            
            if 'rag' in yaml_config:
                rag_config = yaml_config['rag']
                config.n_retrieve = rag_config.get('retrieve_count', config.n_retrieve)
                config.similarity_threshold = rag_config.get('similarity_threshold', config.similarity_threshold)
                config.no_rag = rag_config.get('no_rag', config.no_rag)
            
            if 'ui' in yaml_config:
                ui_config = yaml_config['ui']
                config.verbose = ui_config.get('verbose', config.verbose)
                config.interactive = ui_config.get('interactive', config.interactive)
                config.system_prompt = ui_config.get('system_prompt', config.system_prompt)
                config.chat_template = ui_config.get('chat_template', config.chat_template)
            
            if 'performance' in yaml_config:
                perf_config = yaml_config['performance']
                config.batch_size = perf_config.get('batch_size', config.batch_size)
                config.use_mlock = perf_config.get('use_mlock', config.use_mlock)
                config.use_mmap = perf_config.get('use_mmap', config.use_mmap)
            
            logging.info(f"Loaded configuration from {yaml_path}")
            
        except Exception as e:
            logging.error(f"Error loading config from {yaml_path}: {e}")
            logging.info("Using default configuration")
        
        return config
    
    def update_from_args(self, args):
        """Update configuration with command-line arguments."""
        # Only update if the argument was explicitly provided (not None)
        if args.model_path is not None:
            self.model_path = args.model_path
        if args.folder_path is not None:
            self.folder_path = args.folder_path
        if args.chunk_size != 1000:  # Default value check
            self.chunk_size = args.chunk_size
        if args.chunk_overlap != 200:  # Default value check
            self.chunk_overlap = args.chunk_overlap
        if args.embedding_model != "all-MiniLM-L6-v2":  # Default value check
            self.embedding_model = args.embedding_model
        if args.n_ctx != 4096:  # Default value check
            self.n_ctx = args.n_ctx
        if args.temperature != 0.7:  # Default value check
            self.temperature = args.temperature
        if args.max_tokens != 2048:  # Default value check
            self.max_tokens = args.max_tokens
        if args.n_retrieve != 5:  # Default value check
            self.n_retrieve = args.n_retrieve
        if hasattr(args, 'no_rag') and args.no_rag:
            self.no_rag = args.no_rag
        if args.verbose:
            self.verbose = args.verbose
        if hasattr(args, 'system_prompt') and args.system_prompt:
            self.system_prompt = args.system_prompt
        if hasattr(args, 'chat_template') and args.chat_template != "auto":
            self.chat_template = args.chat_template


class DocuChat:
    """Main DocuChat application class."""
    
    def __init__(self, config: DocuChatConfig):
        self.config = config
        self.llm = None
        self.vector_store = None
        self.document_processor = None
        
        # Setup logging
        log_level = logging.DEBUG if config.verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def initialize(self):
        """Initialize all components."""
        logging.info("Initializing DocuChat...")
        
        # Initialize document processor
        self.document_processor = DocumentProcessor(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        
        # Initialize vector store
        self.vector_store = VectorStore(
            collection_name=self.config.collection_name,
            embedding_model=self.config.embedding_model
        )
        
        # Load documents if folder path is provided
        if self.config.folder_path:
            logging.info(f"Loading documents from: {self.config.folder_path}")
            documents = self.document_processor.load_documents(self.config.folder_path)
            if documents:
                self.vector_store.add_documents(documents)
            else:
                logging.warning("No documents were loaded")
        
        # Initialize LLM
        logging.info(f"Loading model: {self.config.model_path}")
        self.llm = Llama(
            model_path=self.config.model_path,
            n_ctx=self.config.n_ctx,
            n_threads=self.config.n_threads,
            n_gpu_layers=self.config.n_gpu_layers,
            use_mlock=self.config.use_mlock,
            use_mmap=self.config.use_mmap,
            verbose=self.config.verbose
        )
        
        logging.info("DocuChat initialized successfully!")
    
    def generate_response(self, query: str, context_docs: List[str]) -> str:
        """Generate response using the LLM with retrieved context."""
        # Prepare context
        context = "\n\n".join(context_docs) if context_docs else "No relevant context found."
        
        # Create prompt
        if self.config.chat_template == "chatml":
            prompt = f"<|im_start|>system\n{self.config.system_prompt}<|im_end|>\n<|im_start|>user\nContext:\n{context}\n\nQuestion: {query}<|im_end|>\n<|im_start|>assistant\n"
        elif self.config.chat_template == "llama2":
            prompt = f"<s>[INST] <<SYS>>\n{self.config.system_prompt}\n<</SYS>>\n\nContext:\n{context}\n\nQuestion: {query} [/INST]"
        elif self.config.chat_template == "alpaca":
            prompt = f"### Instruction:\n{self.config.system_prompt}\n\n### Input:\nContext:\n{context}\n\nQuestion: {query}\n\n### Response:\n"
        else:  # auto or simple
            prompt = f"{self.config.system_prompt}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        
        # Generate response
        response = self.llm(
            prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            repeat_penalty=self.config.repeat_penalty,
            stop=["\n\n", "Question:", "Context:"] if self.config.chat_template == "auto" else None
        )
        
        return response['choices'][0]['text'].strip()
    
    def chat(self, query: str) -> str:
        """Process a chat query and return response."""
        if not self.llm:
            raise RuntimeError("DocuChat not initialized. Call initialize() first.")
        
        # Retrieve relevant documents (skip if no_rag is enabled)
        context_docs = []
        if not self.config.no_rag and self.vector_store and query.strip():
            context_docs = self.vector_store.search(query, n_results=self.config.n_retrieve)
            if self.config.verbose:
                logging.info(f"Retrieved {len(context_docs)} relevant documents")
        elif self.config.no_rag and self.config.verbose:
            logging.info("RAG disabled - using LLM only mode")
        
        # Generate response
        response = self.generate_response(query, context_docs)
        return response
    
    def interactive_chat(self):
        """Start interactive chat session."""
        print(f"\n{Fore.CYAN}🤖 DocuChat Interactive Mode{Style.RESET_ALL}")
        print("Type 'quit', 'exit', or 'bye' to end the conversation.")
        print("Type 'help' for available commands.\n")
        
        while True:
            try:
                user_input = input(f"\n{Fore.BLUE}👤 You: {Style.RESET_ALL}").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print(f"\n{Fore.YELLOW}👋 Goodbye!{Style.RESET_ALL}")
                    break
                
                if user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                print(f"\n{Fore.GREEN}🤖 DocuChat: {Style.RESET_ALL}", end="", flush=True)
                response = self.chat(user_input)
                print(response)
                
            except KeyboardInterrupt:
                print(f"\n\n{Fore.YELLOW}👋 Goodbye!{Style.RESET_ALL}")
                break
            except Exception as e:
                print(f"\n{Fore.RED}❌ Error: {e}{Style.RESET_ALL}")
                if self.config.verbose:
                    import traceback
                    traceback.print_exc()
    
    def _show_help(self):
        """Show help information."""
        print("\n📖 Available commands:")
        print("  help  - Show this help message")
        print("  quit  - Exit the chat")
        print("  exit  - Exit the chat")
        print("  bye   - Exit the chat")
        print("\n💡 Tips:")
        if self.config.no_rag:
            print("  - LLM-only mode enabled (no document retrieval)")
            print("  - Ask any questions directly to the language model")
        else:
            print("  - Ask questions about your documents")
            print("  - Be specific for better results")
            print("  - The AI will use document context to answer")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="DocuChat - Chat with your documents using local LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Use default config file
  python docuchat.py
  
  # Use custom config file
  python docuchat.py --config my_config.yaml
  
  # Override config with command line arguments
  python docuchat.py --model_path ./models/llama-2-7b.gguf --folder_path ./docs
  
  # Command line only (no config file)
  python docuchat.py --config "" --model_path ./model.gguf --folder_path ./docs
  
  # LLM-only mode without RAG
  python docuchat.py --no-rag --model_path ./model.gguf
        """
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to YAML configuration file (default: config/config.yaml)"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the GGUF model file (overrides config file setting)"
    )
    parser.add_argument(
        "--n_ctx",
        type=int,
        default=4096,
        help="Context window size (default: 4096)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate (default: 2048)"
    )
    
    # Document arguments
    parser.add_argument(
        "--folder_path",
        type=str,
        help="Path to folder containing documents"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="Document chunk size (default: 1000)"
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=200,
        help="Chunk overlap size (default: 200)"
    )
    
    # Embedding arguments
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model for embeddings (default: all-MiniLM-L6-v2)"
    )
    
    # RAG arguments
    parser.add_argument(
        "--n_retrieve",
        type=int,
        default=5,
        help="Number of documents to retrieve (default: 5)"
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable RAG (Retrieval-Augmented Generation) and use LLM only"
    )
    
    # Chat arguments
    parser.add_argument(
        "--system_prompt",
        type=str,
        help="System prompt for the assistant"
    )
    parser.add_argument(
        "--chat_template",
        type=str,
        default="auto",
        choices=["auto", "chatml", "llama2", "alpaca"],
        help="Chat template format (default: auto)"
    )
    
    # Other arguments
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Single query to process (non-interactive mode)"
    )
    
    args = parser.parse_args()
    
    # Load configuration from YAML file first
    if args.config and os.path.exists(args.config):
        config = DocuChatConfig.from_yaml(args.config)
    else:
        config = DocuChatConfig()
    
    # Apply command-line argument overrides
    config.update_from_args(args)
    
    # Validate required settings
    if not config.model_path:
        print("Error: Model path must be specified either in config file or via --model_path argument")
        sys.exit(1)
    
    return config, args


def main():
    """Main application entry point."""
    try:
        config, args = parse_arguments()
        
        # Create and initialize DocuChat
        docuchat = DocuChat(config)
        docuchat.initialize()
        
        # Handle single query or interactive mode
        if args.query:
            # Single query mode
            response = docuchat.chat(args.query)
            print(f"\n🤖 Response: {response}")
        else:
            # Interactive mode
            docuchat.interactive_chat()
    
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if '--verbose' in sys.argv:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
