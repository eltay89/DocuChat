#!/usr/bin/env python3
"""
Tests for DocuChat Tools System
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

try:
    from tools.base_tool import BaseTool
    from tools.calculator_tool import CalculatorTool
    from tools.read_file_tool import ReadFileTool
    from tools.write_file_tool import WriteFileTool
    from tools.search_tool import SearchTool
    from tools.task_done_tool import TaskDoneTool
    from tools import discover_tools
except ImportError:
    # Fallback import path
    import sys
    sys.path.append('.')
    from tools.base_tool import BaseTool
    from tools.calculator_tool import CalculatorTool
    from tools.read_file_tool import ReadFileTool
    from tools.write_file_tool import WriteFileTool
    from tools.search_tool import SearchTool
    from tools.task_done_tool import TaskDoneTool
    from tools import discover_tools


class TestBaseTool:
    """Test cases for BaseTool abstract class."""
    
    def test_base_tool_is_abstract(self):
        """Test that BaseTool cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseTool({})


class TestCalculatorTool:
    """Test cases for CalculatorTool."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tool = CalculatorTool({})
    
    def test_tool_properties(self):
        """Test tool basic properties."""
        assert self.tool.name == "calculate"
        assert "mathematical" in self.tool.description.lower()
        assert "expression" in self.tool.parameters["properties"]
    
    def test_simple_calculation(self):
        """Test simple mathematical calculations."""
        result = self.tool.execute({"expression": "2 + 2"})
        assert "4" in str(result)
    
    def test_complex_calculation(self):
        """Test complex mathematical expressions."""
        result = self.tool.execute({"expression": "sqrt(16) + 2 * 3"})
        assert "10" in str(result)
    
    def test_invalid_expression(self):
        """Test handling of invalid expressions."""
        result = self.tool.execute({"expression": "invalid_expression"})
        assert "error" in str(result).lower() or "invalid" in str(result).lower()
    
    def test_security_restrictions(self):
        """Test that dangerous operations are blocked."""
        dangerous_expressions = [
            "__import__('os').system('ls')",
            "exec('print(1)')",
            "eval('1+1')"
        ]
        
        for expr in dangerous_expressions:
            result = self.tool.execute({"expression": expr})
            # Should either error or not execute the dangerous code
            assert "error" in str(result).lower() or "invalid" in str(result).lower()


class TestReadFileTool:
    """Test cases for ReadFileTool."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tool = ReadFileTool({})
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_tool_properties(self):
        """Test tool basic properties."""
        assert self.tool.name == "read_file"
        assert "read" in self.tool.description.lower()
        assert "file_path" in self.tool.parameters["properties"]
    
    def test_read_existing_file(self):
        """Test reading an existing file."""
        # Create test file
        test_file = Path(self.temp_dir) / "test.txt"
        test_content = "Hello, World!"
        test_file.write_text(test_content)
        
        # Read file
        result = self.tool.execute({"file_path": str(test_file)})
        assert test_content in str(result)
    
    def test_read_nonexistent_file(self):
        """Test reading a file that doesn't exist."""
        nonexistent_file = Path(self.temp_dir) / "nonexistent.txt"
        result = self.tool.execute({"file_path": str(nonexistent_file)})
        assert "error" in str(result).lower() or "not found" in str(result).lower()
    
    def test_security_restrictions(self):
        """Test that restricted files cannot be read."""
        restricted_paths = [
            "/etc/passwd",
            "C:\\Windows\\System32\\config\\SAM",
            "../../../etc/passwd"
        ]
        
        for path in restricted_paths:
            result = self.tool.execute({"file_path": path})
            # Should either error or deny access
            assert "error" in str(result).lower() or "access denied" in str(result).lower() or "restricted" in str(result).lower()


class TestWriteFileTool:
    """Test cases for WriteFileTool."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tool = WriteFileTool({})
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_tool_properties(self):
        """Test tool basic properties."""
        assert self.tool.name == "write_file"
        assert "write" in self.tool.description.lower()
        assert "file_path" in self.tool.parameters["properties"]
        assert "content" in self.tool.parameters["properties"]
    
    def test_write_new_file(self):
        """Test writing to a new file."""
        test_file = Path(self.temp_dir) / "new_file.txt"
        test_content = "This is new content."
        
        result = self.tool.execute({
            "file_path": str(test_file),
            "content": test_content
        })
        
        # Check that file was created
        assert test_file.exists()
        assert test_file.read_text() == test_content
        assert "success" in str(result).lower() or "written" in str(result).lower()
    
    def test_overwrite_existing_file(self):
        """Test overwriting an existing file."""
        test_file = Path(self.temp_dir) / "existing_file.txt"
        original_content = "Original content"
        new_content = "New content"
        
        # Create original file
        test_file.write_text(original_content)
        
        # Overwrite file
        result = self.tool.execute({
            "file_path": str(test_file),
            "content": new_content
        })
        
        # Check that file was overwritten
        assert test_file.read_text() == new_content
        assert "success" in str(result).lower() or "written" in str(result).lower()
    
    def test_security_restrictions(self):
        """Test that restricted locations cannot be written to."""
        restricted_paths = [
            "/etc/passwd",
            "C:\\Windows\\System32\\test.txt",
            "/bin/test"
        ]
        
        for path in restricted_paths:
            result = self.tool.execute({
                "file_path": path,
                "content": "test content"
            })
            # Should either error or deny access
            assert "error" in str(result).lower() or "access denied" in str(result).lower() or "restricted" in str(result).lower()


class TestSearchTool:
    """Test cases for SearchTool."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tool = SearchTool({})
    
    def test_tool_properties(self):
        """Test tool basic properties."""
        assert self.tool.name == "search"
        assert "search" in self.tool.description.lower()
        assert "query" in self.tool.parameters["properties"]
    
    @patch('requests.get')
    def test_search_execution(self, mock_get):
        """Test search execution with mocked response."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "<html><body>Test search results</body></html>"
        mock_get.return_value = mock_response
        
        result = self.tool.execute({"query": "test query"})
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_empty_query(self):
        """Test handling of empty search query."""
        result = self.tool.execute({"query": ""})
        assert "error" in str(result).lower() or "empty" in str(result).lower()


class TestTaskDoneTool:
    """Test cases for TaskDoneTool."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tool = TaskDoneTool({})
    
    def test_tool_properties(self):
        """Test tool basic properties."""
        assert self.tool.name == "task_done"
        assert "task" in self.tool.description.lower() or "complete" in self.tool.description.lower()
        assert "message" in self.tool.parameters["properties"]
    
    def test_task_completion(self):
        """Test task completion message."""
        test_message = "Task completed successfully"
        result = self.tool.execute({"message": test_message})
        assert test_message in str(result)


class TestToolDiscovery:
    """Test cases for tool discovery system."""
    
    def test_discover_tools(self):
        """Test that tools are discovered correctly."""
        tools = discover_tools()
        
        # Check that tools were discovered
        assert isinstance(tools, dict)
        assert len(tools) > 0
        
        # Check that expected tools are present
        expected_tools = ["calculate", "read_file", "write_file", "search", "task_done"]
        for tool_name in expected_tools:
            assert tool_name in tools
            assert isinstance(tools[tool_name], BaseTool)
    
    def test_tool_schemas(self):
        """Test that all tools have valid schemas."""
        tools = discover_tools()
        
        for tool_name, tool in tools.items():
            # Check basic properties
            assert hasattr(tool, 'name')
            assert hasattr(tool, 'description')
            assert hasattr(tool, 'parameters')
            assert hasattr(tool, 'execute')
            
            # Check that name matches
            assert tool.name == tool_name
            
            # Check that parameters is a valid schema
            assert isinstance(tool.parameters, dict)
            assert "type" in tool.parameters
            assert "properties" in tool.parameters


if __name__ == "__main__":
    pytest.main([__file__])
