#!/usr/bin/env python3
"""
Tests for DocuChat Vector Store
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

try:
    from src.docuchat.core.vector_store import VectorStore
except ImportError:
    # Fallback import path
    import sys
    sys.path.append('src')
    from docuchat.core.vector_store import VectorStore


class TestVectorStore:
    """Test cases for VectorStore class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.vector_store = VectorStore(
            persist_directory=self.temp_dir,
            collection_name="test_collection"
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test vector store initialization."""
        assert self.vector_store is not None
        assert hasattr(self.vector_store, 'add_documents')
        assert hasattr(self.vector_store, 'similarity_search')
        assert hasattr(self.vector_store, 'collection')
    
    def test_add_single_document(self):
        """Test adding a single document."""
        # Create mock document
        mock_doc = Mock()
        mock_doc.page_content = "This is a test document about artificial intelligence."
        mock_doc.metadata = {"source": "test.txt", "page": 1}
        
        # Add document
        result = self.vector_store.add_documents([mock_doc])
        
        # Verify addition
        assert result is not None
        
        # Check that document can be retrieved
        search_results = self.vector_store.similarity_search("artificial intelligence", top_k=1)
        assert len(search_results) > 0
        assert "artificial intelligence" in search_results[0]['content']
    
    def test_add_multiple_documents(self):
        """Test adding multiple documents."""
        # Create mock documents
        docs = []
        for i in range(3):
            mock_doc = Mock()
            mock_doc.page_content = f"Document {i} about topic {i}"
            mock_doc.metadata = {"source": f"doc{i}.txt", "page": 1}
            docs.append(mock_doc)
        
        # Add documents
        result = self.vector_store.add_documents(docs)
        assert result is not None
        
        # Verify all documents can be found
        for i in range(3):
            search_results = self.vector_store.similarity_search(f"topic {i}", top_k=1)
            assert len(search_results) > 0
    
    def test_similarity_search(self):
        """Test similarity search functionality."""
        # Add test documents
        docs = [
            Mock(page_content="Python programming language", metadata={"source": "python.txt"}),
            Mock(page_content="JavaScript web development", metadata={"source": "js.txt"}),
            Mock(page_content="Machine learning algorithms", metadata={"source": "ml.txt"})
        ]
        
        self.vector_store.add_documents(docs)
        
        # Test search
        results = self.vector_store.similarity_search("programming", top_k=2)
        
        assert len(results) <= 2
        assert len(results) > 0
        
        # Check result structure
        for result in results:
            assert 'content' in result
            assert 'metadata' in result
            assert isinstance(result['content'], str)
            assert isinstance(result['metadata'], dict)
    
    def test_search_with_no_results(self):
        """Test search when no documents match."""
        # Search without adding any documents
        results = self.vector_store.similarity_search("nonexistent topic", top_k=5)
        
        # Should return empty list
        assert isinstance(results, list)
        assert len(results) == 0
    
    def test_search_top_k_parameter(self):
        """Test that top_k parameter works correctly."""
        # Add multiple documents
        docs = []
        for i in range(10):
            mock_doc = Mock()
            mock_doc.page_content = f"Document {i} with similar content"
            mock_doc.metadata = {"source": f"doc{i}.txt"}
            docs.append(mock_doc)
        
        self.vector_store.add_documents(docs)
        
        # Test different top_k values
        for k in [1, 3, 5]:
            results = self.vector_store.similarity_search("similar content", top_k=k)
            assert len(results) <= k
            assert len(results) > 0
    
    def test_persistence(self):
        """Test that vector store persists data."""
        # Add document to first instance
        mock_doc = Mock()
        mock_doc.page_content = "Persistent test document"
        mock_doc.metadata = {"source": "persistent.txt"}
        
        self.vector_store.add_documents([mock_doc])
        
        # Create new instance with same directory
        new_vector_store = VectorStore(
            persist_directory=self.temp_dir,
            collection_name="test_collection"
        )
        
        # Search in new instance
        results = new_vector_store.similarity_search("persistent", top_k=1)
        assert len(results) > 0
        assert "persistent" in results[0]['content'].lower()
    
    def test_empty_document_handling(self):
        """Test handling of empty documents."""
        # Create empty document
        empty_doc = Mock()
        empty_doc.page_content = ""
        empty_doc.metadata = {"source": "empty.txt"}
        
        # Should handle gracefully
        result = self.vector_store.add_documents([empty_doc])
        assert result is not None
    
    def test_document_metadata_preservation(self):
        """Test that document metadata is preserved."""
        # Create document with rich metadata
        mock_doc = Mock()
        mock_doc.page_content = "Document with metadata"
        mock_doc.metadata = {
            "source": "test.txt",
            "page": 1,
            "author": "Test Author",
            "date": "2024-01-01",
            "category": "test"
        }
        
        self.vector_store.add_documents([mock_doc])
        
        # Search and verify metadata
        results = self.vector_store.similarity_search("metadata", top_k=1)
        assert len(results) > 0
        
        result_metadata = results[0]['metadata']
        assert result_metadata['source'] == "test.txt"
        assert result_metadata['author'] == "Test Author"
        assert result_metadata['category'] == "test"
    
    def test_collection_name_isolation(self):
        """Test that different collection names are isolated."""
        # Create second vector store with different collection
        vector_store2 = VectorStore(
            persist_directory=self.temp_dir,
            collection_name="test_collection_2"
        )
        
        # Add different documents to each
        doc1 = Mock(page_content="Document in collection 1", metadata={"source": "doc1.txt"})
        doc2 = Mock(page_content="Document in collection 2", metadata={"source": "doc2.txt"})
        
        self.vector_store.add_documents([doc1])
        vector_store2.add_documents([doc2])
        
        # Search in first collection
        results1 = self.vector_store.similarity_search("collection", top_k=5)
        assert len(results1) == 1
        assert "collection 1" in results1[0]['content']
        
        # Search in second collection
        results2 = vector_store2.similarity_search("collection", top_k=5)
        assert len(results2) == 1
        assert "collection 2" in results2[0]['content']


class TestEnhancedVectorStore:
    """Test cases for EnhancedVectorStore class (if available)."""
    
    def setup_method(self):
        """Set up test fixtures."""
        try:
            from src.docuchat.core.enhanced_vector_store import EnhancedVectorStore
            self.temp_dir = tempfile.mkdtemp()
            self.enhanced_vector_store = EnhancedVectorStore(
                persist_directory=self.temp_dir,
                collection_name="test_enhanced_collection"
            )
            self.enhanced_available = True
        except ImportError:
            self.enhanced_available = False
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.enhanced_available:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @pytest.mark.skipif(not hasattr(TestEnhancedVectorStore, 'enhanced_available') or 
                       not TestEnhancedVectorStore.enhanced_available,
                       reason="Enhanced vector store not available")
    def test_enhanced_initialization(self):
        """Test enhanced vector store initialization."""
        assert self.enhanced_vector_store is not None
        assert hasattr(self.enhanced_vector_store, 'hybrid_search')
        assert hasattr(self.enhanced_vector_store, 'rerank_results')
    
    @pytest.mark.skipif(not hasattr(TestEnhancedVectorStore, 'enhanced_available') or 
                       not TestEnhancedVectorStore.enhanced_available,
                       reason="Enhanced vector store not available")
    def test_hybrid_search(self):
        """Test hybrid search functionality."""
        # Add test documents
        docs = [
            Mock(page_content="Python programming tutorial", metadata={"source": "python.txt"}),
            Mock(page_content="Advanced Python concepts", metadata={"source": "advanced.txt"}),
            Mock(page_content="Web development with JavaScript", metadata={"source": "web.txt"})
        ]
        
        self.enhanced_vector_store.add_documents(docs)
        
        # Test hybrid search
        results = self.enhanced_vector_store.hybrid_search("Python programming", top_k=2)
        
        assert len(results) <= 2
        assert len(results) > 0
        
        # Should find Python-related documents
        python_found = any("python" in result['content'].lower() for result in results)
        assert python_found


if __name__ == "__main__":
    pytest.main([__file__])
