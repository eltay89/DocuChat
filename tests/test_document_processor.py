#!/usr/bin/env python3
"""
Tests for DocuChat Document Processor
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

try:
    from src.docuchat.core.document_processor import DocumentProcessor
except ImportError:
    # Fallback import path
    import sys
    sys.path.append('src')
    from docuchat.core.document_processor import DocumentProcessor


class TestDocumentProcessor:
    """Test cases for DocumentProcessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = DocumentProcessor()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test processor initialization."""
        assert self.processor is not None
        assert hasattr(self.processor, 'process_file')
        assert hasattr(self.processor, 'process_directory')
    
    def test_process_text_file(self):
        """Test processing a simple text file."""
        # Create test file
        test_file = Path(self.temp_dir) / "test.txt"
        test_content = "This is a test document for DocuChat."
        test_file.write_text(test_content)
        
        # Process file
        result = self.processor.process_file(str(test_file))
        
        # Verify result
        assert result is not None
        assert len(result) > 0
        assert any(test_content in chunk.page_content for chunk in result)
    
    def test_process_markdown_file(self):
        """Test processing a markdown file."""
        # Create test file
        test_file = Path(self.temp_dir) / "test.md"
        test_content = "# Test Document\n\nThis is a **test** document."
        test_file.write_text(test_content)
        
        # Process file
        result = self.processor.process_file(str(test_file))
        
        # Verify result
        assert result is not None
        assert len(result) > 0
    
    def test_process_unsupported_file(self):
        """Test processing an unsupported file type."""
        # Create test file
        test_file = Path(self.temp_dir) / "test.xyz"
        test_file.write_text("Some content")
        
        # Process file should handle gracefully
        result = self.processor.process_file(str(test_file))
        
        # Should return empty list or handle gracefully
        assert isinstance(result, list)
    
    def test_process_nonexistent_file(self):
        """Test processing a file that doesn't exist."""
        nonexistent_file = Path(self.temp_dir) / "nonexistent.txt"
        
        # Should handle gracefully
        with pytest.raises((FileNotFoundError, IOError)):
            self.processor.process_file(str(nonexistent_file))
    
    def test_process_directory(self):
        """Test processing a directory of files."""
        # Create test files
        test_files = [
            ("doc1.txt", "First document content"),
            ("doc2.md", "# Second Document\n\nContent here"),
            ("doc3.txt", "Third document content")
        ]
        
        for filename, content in test_files:
            test_file = Path(self.temp_dir) / filename
            test_file.write_text(content)
        
        # Process directory
        result = self.processor.process_directory(self.temp_dir)
        
        # Verify result
        assert result is not None
        assert len(result) > 0
        
        # Should have processed multiple files
        processed_content = [chunk.page_content for chunk in result]
        assert any("First document" in content for content in processed_content)
        assert any("Second Document" in content for content in processed_content)
        assert any("Third document" in content for content in processed_content)
    
    def test_process_empty_directory(self):
        """Test processing an empty directory."""
        empty_dir = Path(self.temp_dir) / "empty"
        empty_dir.mkdir()
        
        # Process empty directory
        result = self.processor.process_directory(str(empty_dir))
        
        # Should return empty list
        assert isinstance(result, list)
        assert len(result) == 0
    
    def test_chunk_large_document(self):
        """Test chunking of large documents."""
        # Create large test file
        test_file = Path(self.temp_dir) / "large.txt"
        large_content = "This is a sentence. " * 1000  # Large content
        test_file.write_text(large_content)
        
        # Process file
        result = self.processor.process_file(str(test_file))
        
        # Should create multiple chunks
        assert len(result) > 1
        
        # Each chunk should have reasonable size
        for chunk in result:
            assert len(chunk.page_content) > 0
            assert len(chunk.page_content) <= 2000  # Reasonable chunk size
    
    def test_metadata_extraction(self):
        """Test that metadata is properly extracted."""
        # Create test file
        test_file = Path(self.temp_dir) / "test.txt"
        test_content = "Test content for metadata extraction"
        test_file.write_text(test_content)
        
        # Process file
        result = self.processor.process_file(str(test_file))
        
        # Verify metadata
        assert len(result) > 0
        chunk = result[0]
        assert hasattr(chunk, 'metadata')
        assert 'source' in chunk.metadata
        assert chunk.metadata['source'] == str(test_file)
    
    def test_error_handling(self):
        """Test error handling for various scenarios."""
        # Test with None input
        result = self.processor.process_file(None)
        assert isinstance(result, list)
        assert len(result) == 0
        
        # Test with empty string
        result = self.processor.process_file("")
        assert isinstance(result, list)
        assert len(result) == 0
    
    def test_supported_formats(self):
        """Test that all supported formats are handled."""
        supported_formats = ['.txt', '.md']
        
        for format_ext in supported_formats:
            # Create test file
            test_file = Path(self.temp_dir) / f"test{format_ext}"
            test_file.write_text("Test content")
            
            # Process file
            result = self.processor.process_file(str(test_file))
            
            # Should handle without error
            assert isinstance(result, list)


if __name__ == "__main__":
    pytest.main([__file__])
