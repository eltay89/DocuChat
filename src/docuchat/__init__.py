#!/usr/bin/env python3
"""
DocuChat v2.0 - AI-Powered Document Chat Application

A modern, intelligent document analysis and conversation system with advanced
RAG (Retrieval-Augmented Generation) capabilities.

Features:
- Multi-format document support (PDF, DOCX, TXT, MD, HTML, PPTX, CSV, JSON, Images)
- OCR integration for image-based documents
- Advanced text processing and chunking
- Modern embedding models with hybrid search
- Cross-encoder reranking for improved relevance
- Interactive CLI and web interfaces
- Extensible tool system
- Real-time file monitoring
- Docker support for easy deployment

Author: DocuChat Contributors
Version: 2.0.0
License: MIT
"""

__version__ = "2.0.0"
__author__ = "DocuChat Contributors"
__license__ = "MIT"
__description__ = "AI-powered document chat application with advanced RAG capabilities"

# Core imports for easy access
try:
    from .core.document_processor import DocumentProcessor
    from .core.vector_store import VectorStore
    from .core.chat_engine import ChatEngine
except ImportError:
    # Handle cases where optional dependencies are not installed
    pass

# Enhanced imports (optional)
try:
    from .core.enhanced_document_processor import EnhancedDocumentProcessor
    from .core.enhanced_vector_store import EnhancedVectorStore
except ImportError:
    # Enhanced features require additional dependencies
    pass

# Version info
version_info = tuple(map(int, __version__.split('.')))

# Package metadata
__all__ = [
    '__version__',
    '__author__',
    '__license__',
    '__description__',
    'version_info',
    'DocumentProcessor',
    'VectorStore', 
    'ChatEngine',
    'EnhancedDocumentProcessor',
    'EnhancedVectorStore',
]
