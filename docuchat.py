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
import warnings
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*telemetry.*")
warnings.filterwarnings("ignore", message=".*posthog.*")
# Suppress the specific BOS token RuntimeWarning since we handle it programmatically
warnings.filterwarnings("ignore", message=".*duplicate leading.*begin_of_text.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*duplicate.*BOS.*")
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
os.environ["CHROMA_PRODUCT_TELEMETRY_IMPL"] = "chromadb.telemetry.product.posthog.Posthog"

# Suppress specific ChromaDB telemetry logging
import logging
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

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
        
        supported_files = []
        for file_path in folder.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in ['.txt', '.pdf', '.docx', '.doc', '.md']:
                supported_files.append(file_path)
        
        if not supported_files:
            logging.warning(f"No supported document files found in {folder_path}")
            return documents
        
        logging.info(f"Found {len(supported_files)} supported files to process")
        
        for file_path in supported_files:
            try:
                logging.info(f"Processing: {file_path.name}")
                content = self._load_file(file_path)
                if content and content.strip():
                    chunks = self._chunk_text(content)
                    documents.extend(chunks)
                    logging.info(f"✓ Loaded {len(chunks)} chunks from {file_path.name}")
                else:
                    logging.warning(f"⚠ No content extracted from {file_path.name}")
            except Exception as e:
                logging.error(f"✗ Failed to load {file_path.name}: {e}")
                continue
        
        logging.info(f"Total document chunks loaded: {len(documents)}")
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
        """Load PDF file with improved error handling."""
        text = ""
        try:
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                
                if reader.is_encrypted:
                    logging.warning(f"PDF {file_path.name} is encrypted and cannot be read")
                    return ""
                
                total_pages = len(reader.pages)
                logging.info(f"Processing PDF with {total_pages} pages")
                
                for i, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text += page_text + "\n"
                    except Exception as e:
                        logging.warning(f"Failed to extract text from page {i+1}: {e}")
                        continue
                
                if not text.strip():
                    logging.warning(f"No text could be extracted from PDF {file_path.name}")
                    
        except Exception as e:
            logging.error(f"Error reading PDF {file_path.name}: {e}")
            raise
            
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
        
        # Resolve embedding model path
        resolved_model_path = self._resolve_embedding_model_path(embedding_model)
        
        # Initialize embedding model
        logging.info(f"Loading embedding model: {resolved_model_path}")
        try:
            self.embedding_model = SentenceTransformer(resolved_model_path)
            logging.info(f"Successfully loaded embedding model: {resolved_model_path}")
        except Exception as e:
            logging.error(f"Failed to load embedding model '{resolved_model_path}': {e}")
            logging.info("Falling back to default model: all-MiniLM-L6-v2")
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            self.embedding_model_name = "all-MiniLM-L6-v2"
        
        # Initialize ChromaDB with telemetry disabled
        self.client = chromadb.Client(Settings(
            allow_reset=True,
            anonymized_telemetry=False
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

        # Clear existing collection by checking if it has documents first
        try:
            existing_count = self.collection.count()
            if existing_count > 0:
                # Get all existing IDs and delete them
                existing_data = self.collection.get()
                if existing_data['ids']:
                    self.collection.delete(ids=existing_data['ids'])
                    logging.info(f"Cleared {existing_count} existing documents from collection")
        except Exception as e:
            logging.warning(f"Could not clear existing documents: {e}")
        
        # Add new documents
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
    
    def _resolve_embedding_model_path(self, model_name: str) -> str:
        """Resolve embedding model path, checking local embeddings folder first."""
        # If it's already an absolute path, use it as-is
        if os.path.isabs(model_name):
            return model_name
        
        # Check if it's a relative path starting with ./ or ../
        if model_name.startswith('./') or model_name.startswith('../'):
            return model_name
        
        # Check if model exists in local embeddings folder
        embeddings_dir = os.path.join(os.path.dirname(__file__), 'embeddings')
        local_model_path = os.path.join(embeddings_dir, model_name)
        
        if os.path.exists(local_model_path):
            logging.info(f"Found local embedding model: {local_model_path}")
            return local_model_path
        
        # Check if it's a path relative to current directory
        if os.path.exists(model_name):
            return model_name
        
        # Otherwise, treat it as a Hugging Face model identifier
        logging.info(f"Using Hugging Face model identifier: {model_name}")
        return model_name


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
        self.chat_format = None  # Chat format for llama-cpp-python
        

        
        # UI and interaction
        self.verbose = False
        self.interactive = True
        self.streaming = True  # Default to streaming enabled
        
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
            

            
            # Load streaming configuration from advanced.experimental section
            if 'advanced' in yaml_config and 'experimental' in yaml_config['advanced']:
                experimental_config = yaml_config['advanced']['experimental']
                config.streaming = experimental_config.get('streaming', config.streaming)
            
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
        
        # Handle streaming configuration
        if hasattr(args, 'no_streaming') and args.no_streaming:
            self.streaming = False
        elif hasattr(args, 'streaming') and args.streaming:
            self.streaming = True


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
        
        # Only initialize document processing components if RAG is enabled
        if not self.config.no_rag:
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
                try:
                    documents = self.document_processor.load_documents(self.config.folder_path)
                    if documents:
                        logging.info(f"Successfully loaded {len(documents)} document chunks")
                        self.vector_store.add_documents(documents)
                        logging.info("Documents added to vector store successfully")
                    else:
                        logging.warning("No documents were loaded - check if documents exist in the folder")
                        print(f"⚠️  Warning: No documents found in {self.config.folder_path}")
                        print("   Supported formats: .txt, .pdf, .docx, .md")
                except Exception as e:
                    logging.error(f"Failed to load documents: {e}")
                    print(f"❌ Error loading documents: {e}")
                    print("   Continuing in LLM-only mode...")
                    self.config.no_rag = True
        else:
            logging.info("RAG disabled - skipping document processing and embedding model initialization")
        
        # Initialize LLM with chat format
        logging.info(f"Loading model: {self.config.model_path}")
        
        # Determine chat format based on config
        chat_format = self.config.chat_format
        if not chat_format and self.config.chat_template != "auto":
            # Map chat_template to chat_format
            template_mapping = {
                "chatml": "chatml",
                "llama2": "llama-2", 
                "alpaca": "alpaca"
            }
            chat_format = template_mapping.get(self.config.chat_template)
        
        # Initialize Llama with chat_format parameter
        llm_kwargs = {
            "model_path": self.config.model_path,
            "n_ctx": self.config.n_ctx,
            "n_gpu_layers": self.config.n_gpu_layers,
            "use_mlock": self.config.use_mlock,
            "use_mmap": self.config.use_mmap,
            "verbose": self.config.verbose
        }
        
        if self.config.n_threads:
            llm_kwargs["n_threads"] = self.config.n_threads
            
        if chat_format:
            llm_kwargs["chat_format"] = chat_format
            logging.info(f"Using chat format: {chat_format}")
        
        self.llm = Llama(**llm_kwargs)
        

        
        logging.info("DocuChat initialized successfully!")
    

    

    

    
    def _create_messages(self, query: str, context_docs: List[str]) -> List[Dict[str, str]]:
        """Create messages array for chat completion API."""
        messages = []
        
        # Create system message
        if self.config.no_rag or not context_docs:
            system_content = "You are a helpful AI assistant. Answer questions using your knowledge."
        else:
            context = "\n\n".join(context_docs)
            system_content = f"{self.config.system_prompt}\n\nContext:\n{context}"
        
        messages.append({"role": "system", "content": system_content})
        
        # Add user message
        if self.config.no_rag or not context_docs:
            user_content = query
        else:
            user_content = f"Question: {query}"
        
        messages.append({"role": "user", "content": user_content})
        
        return messages
     

    
    def generate_response(self, query: str, context_docs: List[str]) -> str:
        """Generate response using the LLM with retrieved context."""
        # Create messages for chat completion
        messages = self._create_messages(query, context_docs)
        
        # Generate response using create_chat_completion
        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            repeat_penalty=self.config.repeat_penalty,
            stop=["Question:", "Context:"] if context_docs else None
        )
        
        return response['choices'][0]['message']['content'].strip()
    
    def generate_response_streaming(self, query: str, context_docs: List[str]) -> str:
        """Generate response using the LLM with retrieved context and streaming output."""
        # Create messages for chat completion
        messages = self._create_messages(query, context_docs)
        
        # Generate response with streaming using create_chat_completion
        full_response = ""
        
        stream = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            repeat_penalty=self.config.repeat_penalty,
            stop=["Question:", "Context:"] if context_docs else None,
            stream=True
        )
        
        for output in stream:
            # Handle both delta and content formats
            if 'delta' in output['choices'][0] and 'content' in output['choices'][0]['delta']:
                token = output['choices'][0]['delta']['content']
            elif 'message' in output['choices'][0] and 'content' in output['choices'][0]['message']:
                token = output['choices'][0]['message']['content']
            else:
                continue
                
            if token is None:
                continue
                
            full_response += token
            print(token, end='', flush=True)
        
        return full_response.strip()
    
    def chat(self, query: str) -> str:
        """Process a chat query and return response."""
        if not self.llm:
            raise RuntimeError("DocuChat not initialized. Call initialize() first.")
        
        # Retrieve relevant documents (skip if no_rag is enabled)
        context_docs = []
        docs_used = False
        if not self.config.no_rag and self.vector_store and query.strip():
            try:
                context_docs = self.vector_store.search(query, n_results=self.config.n_retrieve)
                if context_docs:
                    docs_used = True
                    # Clear the searching indicator and show found documents
                    print(f"\r{Fore.GREEN}📄 Found {len(context_docs)} relevant document sections{Style.RESET_ALL}")
                    if self.config.verbose:
                        logging.info(f"Retrieved {len(context_docs)} relevant documents")
                        logging.info("Using document context for response")
                else:
                    print(f"\r{Fore.YELLOW}📄 No relevant documents found, using general knowledge{Style.RESET_ALL}")
            except Exception as e:
                print(f"\r{Fore.YELLOW}⚠️  Document search failed, using LLM-only mode: {e}{Style.RESET_ALL}")
                logging.error(f"Error during document search: {e}")
                logging.info("Falling back to LLM-only mode for this query")
                context_docs = []
        elif self.config.no_rag and self.config.verbose:
            logging.info("RAG disabled - using LLM only mode")
        
        # Generate response based on streaming configuration
        if self.config.streaming:
            response = self.generate_response_streaming(query, context_docs)
        else:
            response = self.generate_response(query, context_docs)
            print(response, end="", flush=True)
        
        # Add a subtle indicator if documents were used
        if docs_used and not self.config.verbose:
            print(f"\n\n{Fore.CYAN}💡 Response based on your documents{Style.RESET_ALL}")
        
        return response
    
    def interactive_chat(self):
        """Start interactive chat session."""
        print(f"\n{Fore.CYAN}🤖 DocuChat Interactive Mode{Style.RESET_ALL}")
        print("Type 'quit', 'exit', or 'bye' to end the conversation.")
        print("Type 'help' for available commands.")
        print("Type 'status' to check document loading status.\n")
        
        # Show initial status
        self._show_status()
        
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
                    
                if user_input.lower() == 'status':
                    self._show_status()
                    continue
                
                # Show thinking indicator for document retrieval
                if not self.config.no_rag and self.vector_store:
                    print(f"\n{Fore.CYAN}🔍 Searching documents...{Style.RESET_ALL}", end="", flush=True)
                
                print(f"\n{Fore.GREEN}🤖 DocuChat: {Style.RESET_ALL}", end="", flush=True)
                response = self.chat(user_input)
                # Response is already printed by streaming, just add a newline
                print()
                
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
        print("  help   - Show this help message")
        print("  status - Show document loading status")
        print("  quit   - Exit the chat")
        print("  exit   - Exit the chat")
        print("  bye    - Exit the chat")
        print("\n💡 Tips:")
        if self.config.no_rag:
            print("  - LLM-only mode enabled (no document retrieval)")
            print("  - Ask any questions directly to the language model")
        else:
            print("  - Ask questions about your documents")
            print("  - Be specific for better results")
            print("  - The AI will use document context to answer")
            
    def _show_status(self):
        """Show current system status."""
        print(f"\n📊 {Fore.CYAN}System Status:{Style.RESET_ALL}")
        print(f"  Model: {self.config.model_path}")
        

        
        if self.config.no_rag:
            print(f"  Mode: {Fore.YELLOW}LLM-only (no document retrieval){Style.RESET_ALL}")
        else:
            print(f"  Documents folder: {self.config.folder_path}")
            try:
                # Check if vector store has documents
                if self.vector_store and hasattr(self.vector_store, 'collection'):
                    count = self.vector_store.collection.count()
                    if count > 0:
                        print(f"  Mode: {Fore.GREEN}RAG enabled{Style.RESET_ALL}")
                        print(f"  Documents loaded: {Fore.GREEN}{count} chunks{Style.RESET_ALL}")
                    else:
                        print(f"  Mode: {Fore.YELLOW}RAG enabled but no documents loaded{Style.RESET_ALL}")
                        print(f"  Documents loaded: {Fore.RED}0 chunks{Style.RESET_ALL}")
                else:
                    print(f"  Mode: {Fore.RED}Vector store not initialized{Style.RESET_ALL}")
            except Exception as e:
                print(f"  Mode: {Fore.RED}Error checking document status: {e}{Style.RESET_ALL}")
            
            print(f"  Embedding model: {self.config.embedding_model}")
            print(f"  Retrieval count: {self.config.n_retrieve}")
        
        print(f"  Streaming: {Fore.GREEN if self.config.streaming else Fore.RED}{'Enabled' if self.config.streaming else 'Disabled'}{Style.RESET_ALL}")


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
  
  # Download an embedding model from Hugging Face
  python docuchat.py --download_embedding_model sentence-transformers/all-mpnet-base-v2
  
  # Use custom embedding model from embeddings folder
  python docuchat.py --embedding_model sentence-transformers--all-mpnet-base-v2
  
  # Use Hugging Face embedding model directly
  python docuchat.py --embedding_model sentence-transformers/all-mpnet-base-v2
  
  # Use local embedding model with path
  python docuchat.py --embedding_model ./embeddings/multilingual-model
  
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
        "--model_path", "--model-path",
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
        help="Embedding model for document similarity. Options: \n"
             "  - Hugging Face model ID (e.g., 'all-MiniLM-L6-v2', 'all-mpnet-base-v2')\n"
             "  - Local model in embeddings/ folder (e.g., 'my-custom-model')\n"
             "  - Relative/absolute path (e.g., './embeddings/model', '/path/to/model')\n"
             "  Default: all-MiniLM-L6-v2"
    )
    parser.add_argument(
        "--download_embedding_model",
        type=str,
        help="Download an embedding model from Hugging Face to the embeddings/ folder. \n"
             "Specify the Hugging Face model ID (e.g., 'sentence-transformers/all-mpnet-base-v2')"
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
        "--streaming",
        action="store_true",
        help="Enable streaming output (default: True, use --no-streaming to disable)"
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable streaming output"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Single query to process (non-interactive mode)"
    )
    
    args = parser.parse_args()
    
    # Handle embedding model download early (before validation)
    if hasattr(args, 'download_embedding_model') and args.download_embedding_model:
        download_embedding_model(args.download_embedding_model)
        sys.exit(0)
    
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


def download_embedding_model(model_id: str, embeddings_dir: str = "embeddings") -> str:
    """Download an embedding model from Hugging Face to the embeddings folder.
    
    Args:
        model_id: Hugging Face model ID (e.g., 'sentence-transformers/all-mpnet-base-v2')
        embeddings_dir: Directory to save the model (default: 'embeddings')
        
    Returns:
        str: Path to the downloaded model
    """
    try:
        from sentence_transformers import SentenceTransformer
        import os
        
        # Create embeddings directory if it doesn't exist
        os.makedirs(embeddings_dir, exist_ok=True)
        
        # Extract model name from model_id for local folder name
        model_name = model_id.replace('/', '--')
        local_path = os.path.join(embeddings_dir, model_name)
        
        print(f"📥 Downloading embedding model '{model_id}' to '{local_path}'...")
        
        # Download and save the model
        model = SentenceTransformer(model_id)
        model.save(local_path)
        
        print(f"✅ Successfully downloaded embedding model to: {local_path}")
        print(f"💡 You can now use it with: --embedding_model {model_name}")
        
        return local_path
        
    except Exception as e:
        print(f"❌ Error downloading embedding model '{model_id}': {e}")
        raise


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
