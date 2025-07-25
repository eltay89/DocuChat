#!/usr/bin/env python3
"""
DocuChat Core Modules

This package contains the core functionality for DocuChat v2.0:
- Document processing and parsing
- Vector storage and retrieval
- Chat engine and conversation management
- Enhanced features (OCR, hybrid search, reranking)
"""

__version__ = "2.0.0"

# Core imports
try:
    from .document_processor import DocumentProcessor
    from .vector_store import VectorStore
    from .chat_engine import ChatEngine
except ImportError as e:
    print(f"Warning: Could not import core modules: {e}")

# Enhanced imports (optional)
try:
    from .enhanced_document_processor import EnhancedDocumentProcessor
    from .enhanced_vector_store import EnhancedVectorStore
except ImportError:
    # Enhanced features require additional dependencies
    pass

__all__ = [
    'DocumentProcessor',
    'VectorStore',
    'ChatEngine',
    'EnhancedDocumentProcessor',
    'EnhancedVectorStore',
]
