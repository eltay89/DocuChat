#!/usr/bin/env python3
"""
DocuChat Tool System

Extensible tool system for DocuChat v2.0 that allows AI agents to perform
various actions like reading files, searching, calculations, and more.
"""

import os
import importlib
import inspect
from typing import List, Dict, Type
from .base_tool import BaseTool


def discover_tools() -> Dict[str, Type[BaseTool]]:
    """
    Automatically discover and load all available tools.
    
    Returns:
        Dict[str, Type[BaseTool]]: Dictionary mapping tool names to tool classes
    """
    tools = {}
    
    # Get the directory containing this file
    tools_dir = os.path.dirname(__file__)
    
    # Iterate through all Python files in the tools directory
    for filename in os.listdir(tools_dir):
        if filename.endswith('_tool.py') and filename != 'base_tool.py':
            module_name = filename[:-3]  # Remove .py extension
            
            try:
                # Import the module
                module = importlib.import_module(f'.{module_name}', package=__name__)
                
                # Find all classes that inherit from BaseTool
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (issubclass(obj, BaseTool) and 
                        obj != BaseTool and 
                        not inspect.isabstract(obj)):
                        
                        # Create an instance to get the tool name
                        tool_instance = obj()
                        tools[tool_instance.name] = obj
                        
            except ImportError as e:
                print(f"Warning: Could not import tool module {module_name}: {e}")
            except Exception as e:
                print(f"Warning: Error loading tool from {module_name}: {e}")
    
    return tools


def get_available_tools() -> List[BaseTool]:
    """
    Get instances of all available tools.
    
    Returns:
        List[BaseTool]: List of instantiated tool objects
    """
    tool_classes = discover_tools()
    return [tool_class() for tool_class in tool_classes.values()]


def get_tool_by_name(name: str) -> BaseTool:
    """
    Get a specific tool by name.
    
    Args:
        name: The name of the tool to retrieve
        
    Returns:
        BaseTool: The requested tool instance
        
    Raises:
        ValueError: If the tool is not found
    """
    tool_classes = discover_tools()
    
    if name not in tool_classes:
        available_tools = list(tool_classes.keys())
        raise ValueError(f"Tool '{name}' not found. Available tools: {available_tools}")
    
    return tool_classes[name]()


def get_tools_schema() -> List[Dict]:
    """
    Get OpenRouter function schemas for all available tools.
    
    Returns:
        List[Dict]: List of tool schemas for OpenRouter function calling
    """
    tools = get_available_tools()
    return [tool.to_openrouter_schema() for tool in tools]


# Auto-discover tools on import
_available_tools = discover_tools()

__all__ = [
    'BaseTool',
    'discover_tools',
    'get_available_tools', 
    'get_tool_by_name',
    'get_tools_schema',
]

# Export tool classes for direct import
for tool_name, tool_class in _available_tools.items():
    globals()[tool_class.__name__] = tool_class
    __all__.append(tool_class.__name__)
