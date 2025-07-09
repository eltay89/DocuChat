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
from typing import List, Dict, Optional, Any, Tuple, Union, Set
import yaml
import warnings
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import hashlib
import json
from datetime import datetime

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
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_documents(self, folder_path: Union[str, Path]) -> List[str]:
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


class FileMonitor:
    """Monitors document folder for changes and manages file tracking."""
    
    def __init__(self, folder_path: Union[str, Path], cache_dir: str = "./vectordbs") -> None:
        self.folder_path = Path(folder_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # File to store file metadata
        self.metadata_file = self.cache_dir / "file_metadata.json"
        self.file_metadata = self._load_metadata()
        
        # Supported file extensions
        self.supported_extensions = {'.txt', '.pdf', '.docx', '.doc', '.md'}
    
    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load file metadata from cache."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Convert path strings back to Path objects
                    for rel_path, metadata in data.items():
                        if 'path' in metadata:
                            metadata['path'] = Path(metadata['path'])
                    return data
            except Exception as e:
                logging.warning(f"Failed to load file metadata: {e}")
        return {}
    
    def _save_metadata(self) -> None:
        """Save file metadata to cache."""
        try:
            # Convert Path objects to strings for JSON serialization
            serializable_metadata = {}
            for rel_path, data in self.file_metadata.items():
                serializable_metadata[rel_path] = {
                    'path': str(data['path']),
                    'hash': data['hash'],
                    'mtime': data['mtime'],
                    'last_processed': data['last_processed']
                }
            
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_metadata, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save file metadata: {e}")
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file content."""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logging.error(f"Failed to calculate hash for {file_path}: {e}")
            return ""
    
    def scan_for_changes(self) -> Dict[str, List[Path]]:
        """Scan folder for file changes and return categorized changes."""
        if not self.folder_path.exists():
            logging.warning(f"Document folder does not exist: {self.folder_path}")
            return {'added': [], 'modified': [], 'removed': []}
        
        current_files = {}
        changes = {'added': [], 'modified': [], 'removed': []}
        
        # Scan current files
        for file_path in self.folder_path.rglob('*'):
            if (file_path.is_file() and 
                file_path.suffix.lower() in self.supported_extensions):
                
                rel_path = str(file_path.relative_to(self.folder_path))
                file_hash = self._get_file_hash(file_path)
                file_mtime = file_path.stat().st_mtime
                
                current_files[rel_path] = {
                    'path': file_path,
                    'hash': file_hash,
                    'mtime': file_mtime,
                    'last_processed': datetime.now().isoformat()
                }
                
                # Check if file is new or modified
                if rel_path not in self.file_metadata:
                    changes['added'].append(file_path)
                    logging.info(f"New file detected: {rel_path}")
                elif (self.file_metadata[rel_path]['hash'] != file_hash or
                      self.file_metadata[rel_path]['mtime'] != file_mtime):
                    changes['modified'].append(file_path)
                    logging.info(f"Modified file detected: {rel_path}")
        
        # Check for removed files
        for rel_path in self.file_metadata:
            if rel_path not in current_files:
                changes['removed'].append(rel_path)
                logging.info(f"Removed file detected: {rel_path}")
        
        # Update metadata
        self.file_metadata = current_files
        self._save_metadata()
        
        return changes
    
    def get_all_files(self) -> List[Path]:
        """Get list of all tracked files."""
        return [data['path'] for data in self.file_metadata.values() 
                if data['path'].exists()]
    
    def update_processed_files(self, file_paths: List[Path]) -> None:
        """Update the processed timestamp for files that have been successfully processed."""
        current_time = datetime.now().isoformat()
        
        for file_path in file_paths:
            if file_path.exists():
                file_hash = self._get_file_hash(file_path)
                rel_path = str(file_path.relative_to(self.folder_path))
                self.file_metadata[rel_path] = {
                    'path': file_path,
                    'hash': file_hash,
                    'mtime': file_path.stat().st_mtime,
                    'last_processed': current_time
                }
        
        self._save_metadata()
        logging.debug(f"Updated processed timestamp for {len(file_paths)} files")
    
    def clear_cache(self) -> None:
        """Clear all cached metadata."""
        self.file_metadata = {}
        if self.metadata_file.exists():
            self.metadata_file.unlink()
        logging.info("File monitoring cache cleared")


class VectorStore:
    """Manages vector embeddings and similarity search using ChromaDB with persistent storage."""
    
    def __init__(self, collection_name: str = "documents", embedding_model: str = "all-MiniLM-L6-v2", 
                 persist_directory: str = "./vectordbs") -> None:
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
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
        
        # Initialize ChromaDB with persistent storage
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                allow_reset=True,
                anonymized_telemetry=False
            )
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(collection_name)
            logging.info(f"Loaded existing collection: {collection_name} with {self.collection.count()} documents")
        except:
            self.collection = self.client.create_collection(collection_name)
            logging.info(f"Created new collection: {collection_name}")
        
        # Document metadata cache
        self.doc_metadata_file = self.persist_directory / "doc_metadata.json"
        self.doc_metadata = self._load_doc_metadata()
    
    def _load_doc_metadata(self) -> Dict[str, Any]:
        """Load document metadata from cache."""
        if self.doc_metadata_file.exists():
            try:
                with open(self.doc_metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Failed to load document metadata: {e}")
        return {'next_id': 0, 'file_to_ids': {}, 'id_to_file': {}}
    
    def _save_doc_metadata(self) -> None:
        """Save document metadata to cache."""
        try:
            with open(self.doc_metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.doc_metadata, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save document metadata: {e}")
    
    def add_documents(self, documents: List[str], file_path: Optional[str] = None) -> None:
        """Add documents to the vector store with optional file tracking."""
        if not documents:
            logging.warning("No documents to add")
            return

        logging.info(f"Generating embeddings for {len(documents)} documents...")
        embeddings = self.embedding_model.encode(documents, show_progress_bar=True)

        # Generate unique IDs for new documents
        start_id = self.doc_metadata['next_id']
        ids = [f"doc_{start_id + i}" for i in range(len(documents))]
        self.doc_metadata['next_id'] = start_id + len(documents)
        
        # Track file associations if provided
        if file_path:
            self.doc_metadata['file_to_ids'][file_path] = ids
            for doc_id in ids:
                self.doc_metadata['id_to_file'][doc_id] = file_path
        
        # Add new documents to collection
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            ids=ids
        )
        
        # Save metadata
        self._save_doc_metadata()

        logging.info(f"Added {len(documents)} documents to vector store")
    
    def remove_documents_by_file(self, file_path: str) -> None:
        """Remove all documents associated with a specific file."""
        if file_path not in self.doc_metadata['file_to_ids']:
            logging.warning(f"No documents found for file: {file_path}")
            return
        
        doc_ids = self.doc_metadata['file_to_ids'][file_path]
        
        try:
            # Remove from ChromaDB collection
            self.collection.delete(ids=doc_ids)
            
            # Update metadata
            del self.doc_metadata['file_to_ids'][file_path]
            for doc_id in doc_ids:
                if doc_id in self.doc_metadata['id_to_file']:
                    del self.doc_metadata['id_to_file'][doc_id]
            
            self._save_doc_metadata()
            logging.info(f"Removed {len(doc_ids)} documents for file: {file_path}")
            
        except Exception as e:
            logging.error(f"Failed to remove documents for file {file_path}: {e}")
    
    def update_documents_for_file(self, file_path: str, documents: List[str]) -> None:
        """Update documents for a specific file (remove old, add new)."""
        # Remove existing documents for this file
        self.remove_documents_by_file(file_path)
        
        # Add new documents
        if documents:
            self.add_documents(documents, file_path)
    
    def clear_all_documents(self) -> None:
        """Clear all documents from the vector store."""
        try:
            existing_data = self.collection.get()
            if existing_data['ids']:
                self.collection.delete(ids=existing_data['ids'])
                logging.info(f"Cleared {len(existing_data['ids'])} documents from collection")
            
            # Reset metadata
            self.doc_metadata = {'next_id': 0, 'file_to_ids': {}, 'id_to_file': {}}
            self._save_doc_metadata()
            
        except Exception as e:
            logging.error(f"Failed to clear documents: {e}")
    
    def get_document_count(self) -> int:
        """Get total number of documents in the vector store."""
        try:
            return self.collection.count()
        except Exception as e:
            logging.error(f"Failed to get document count: {e}")
            return 0
    
    def search(self, query: str, n_results: int = 5, return_metadata: bool = False):
        """Search for similar documents.
        
        Args:
            query: Search query string
            n_results: Number of results to return
            return_metadata: If True, return (documents, metadata) tuple
            
        Returns:
            List[str] if return_metadata=False, else Tuple[List[str], List[dict]]
        """
        query_embedding = self.embedding_model.encode([query])
        
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results,
            include=['documents', 'metadatas'] if return_metadata else ['documents']
        )
        
        documents = results['documents'][0] if results['documents'] else []
        
        if return_metadata:
            metadata = results.get('metadatas', [[]])[0] if results.get('metadatas') else []
            return documents, metadata
        else:
            return documents
    
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
    
    def __init__(self) -> None:
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
        
        # Logging configuration
        self.log_level = "INFO"
        self.log_file = None
    
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
                config.streaming = ui_config.get('streaming', config.streaming)
            

            
            # Load streaming configuration from advanced.experimental section
            if 'advanced' in yaml_config and 'experimental' in yaml_config['advanced']:
                experimental_config = yaml_config['advanced']['experimental']
                config.streaming = experimental_config.get('streaming', config.streaming)
            
            if 'performance' in yaml_config:
                perf_config = yaml_config['performance']
                config.batch_size = perf_config.get('batch_size', config.batch_size)
                config.use_mlock = perf_config.get('use_mlock', config.use_mlock)
                config.use_mmap = perf_config.get('use_mmap', config.use_mmap)
            
            # Load logging configuration
            if 'logging' in yaml_config:
                log_config = yaml_config['logging']
                config.log_level = log_config.get('level', config.log_level)
                config.log_file = log_config.get('file', config.log_file)
            
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
        self.file_monitor = None
        
        # Setup logging
        log_level = getattr(logging, config.log_level.upper(), logging.INFO)
        if config.verbose:
            log_level = logging.DEBUG
        
        log_config = {
            'level': log_level,
            'format': '%(asctime)s - %(levelname)s - %(message)s'
        }
        
        if config.log_file:
            log_config['filename'] = config.log_file
            log_config['filemode'] = 'a'
        
        logging.basicConfig(**log_config)
    
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
            
            # Initialize file monitor and load documents if folder path is provided
            if self.config.folder_path:
                logging.info(f"Loading documents from: {self.config.folder_path}")
                try:
                    # Initialize file monitor
                    self.file_monitor = FileMonitor(self.config.folder_path)
                    
                    # Check for file changes and update documents
                    self._update_documents_from_folder()
                    
                    doc_count = self.vector_store.get_document_count()
                    if doc_count > 0:
                        logging.info(f"Successfully loaded {doc_count} document chunks")
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
    
    def _update_documents_from_folder(self):
        """Update documents in vector store based on file changes."""
        if not self.file_monitor:
            return
        
        # Check for file changes
        changes = self.file_monitor.scan_for_changes()
        
        # Process removed files
        for file_path in changes['removed']:
            rel_path = str(file_path.relative_to(Path(self.config.folder_path)))
            logging.info(f"Removing documents for deleted file: {rel_path}")
            self.vector_store.remove_documents_by_file(rel_path)
        
        # Process added and modified files
        files_to_process = changes['added'] + changes['modified']
        
        if files_to_process:
            logging.info(f"Processing {len(files_to_process)} new/modified files")
            
            for file_path in files_to_process:
                try:
                    # Load and process the document content
                    content = self.document_processor._load_file(file_path)
                    if content:
                        chunks = self.document_processor._chunk_text(content)
                        
                        if chunks:
                            # Update documents for this file using relative path
                            rel_path = str(file_path.relative_to(Path(self.config.folder_path)))
                            self.vector_store.update_documents_for_file(rel_path, chunks)
                            logging.info(f"Updated {file_path.name}: {len(chunks)} chunks")
                        else:
                            logging.warning(f"No chunks extracted from {file_path.name}")
                    else:
                        logging.warning(f"No content loaded from {file_path.name}")
                        
                except Exception as e:
                    logging.error(f"Failed to process {file_path.name}: {e}")
        
        # Update the file monitor cache for processed files
        if files_to_process:
            self.file_monitor.update_processed_files(files_to_process)
        
        # Log current status
        total_docs = self.vector_store.get_document_count()
        logging.info(f"Vector store now contains {total_docs} document chunks")
    
    def refresh_documents(self, force: bool = False):
        """Manually refresh documents from the folder."""
        if not self.file_monitor or not self.vector_store:
            logging.warning("File monitoring or vector store not initialized")
            return
        
        if force:
            logging.info("Force refresh: clearing all cached metadata")
            self.file_monitor.clear_cache()
            self.vector_store.clear_all_documents()
        
        logging.info("Refreshing documents from folder...")
        self._update_documents_from_folder()
        
        total_docs = self.vector_store.get_document_count()
        logging.info(f"Refresh complete. Vector store contains {total_docs} document chunks")
    

    

    

    
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
        source_files = []
        docs_used = False
        if not self.config.no_rag and self.vector_store and query.strip():
            try:
                search_results = self.vector_store.search(query, n_results=self.config.n_retrieve, return_metadata=True)
                if search_results:
                    if isinstance(search_results, tuple) and len(search_results) == 2:
                        context_docs, metadata_list = search_results
                        # Extract source files from metadata
                        source_files = [meta.get('source', 'Unknown') for meta in metadata_list if meta]
                    else:
                        context_docs = search_results
                        source_files = []
                    
                    docs_used = True
                    # Clear the searching indicator and show found documents
                    print(f"\r{Fore.GREEN}📄 Found {len(context_docs)} relevant document sections{Style.RESET_ALL}")
                    if self.config.verbose:
                        logging.info(f"Retrieved {len(context_docs)} relevant documents")
                        logging.info("Using document context for response")
                        if source_files:
                            unique_sources = list(set(source_files))
                            logging.info(f"Sources: {', '.join(unique_sources)}")
                else:
                    print(f"\r{Fore.YELLOW}📄 No relevant documents found, using general knowledge{Style.RESET_ALL}")
            except Exception as e:
                print(f"\r{Fore.YELLOW}⚠️  Document search failed, using LLM-only mode: {e}{Style.RESET_ALL}")
                logging.error(f"Error during document search: {e}")
                logging.info("Falling back to LLM-only mode for this query")
                context_docs = []
                source_files = []
        elif self.config.no_rag and self.config.verbose:
            logging.info("RAG disabled - using LLM only mode")
        
        # Use the original query
        final_query = query
        
        # Generate response based on streaming configuration
        if self.config.streaming:
            response = self.generate_response_streaming(final_query, context_docs)
        else:
            response = self.generate_response(final_query, context_docs)
            print(response, end="", flush=True)
        
        # Add indicators for data sources used
        if docs_used and not self.config.verbose:
            if source_files:
                unique_sources = list(set(source_files))
                if len(unique_sources) == 1:
                    print(f"\n\n💡 {Fore.CYAN}📄 Based on: {unique_sources[0]}{Style.RESET_ALL}")
                else:
                    print(f"\n\n💡 {Fore.CYAN}📄 Based on {len(unique_sources)} documents: {', '.join(unique_sources[:3])}{'...' if len(unique_sources) > 3 else ''}{Style.RESET_ALL}")
            else:
                print(f"\n\n💡 {Fore.CYAN}📄 Based on your documents{Style.RESET_ALL}")
        
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
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refresh document embeddings by checking for new/modified/removed files"
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force complete refresh by clearing all cached data and re-processing all documents"
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
        
        # Handle refresh operations
        if hasattr(args, 'force_refresh') and args.force_refresh:
            print("🔄 Force refreshing all documents...")
            docuchat.refresh_documents(force=True)
            print("✅ Force refresh completed!")
        elif hasattr(args, 'refresh') and args.refresh:
            print("🔄 Refreshing documents...")
            docuchat.refresh_documents(force=False)
            print("✅ Refresh completed!")
        
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
    finally:
        if 'docuchat' in locals() and docuchat.llm:
            print("Closing Llama model...")
            docuchat.llm.close()
            print("Llama model closed.")
        if '--verbose' in sys.argv:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
