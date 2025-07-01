#!/usr/bin/env python3
"""
DocuChat Configuration Examples

This file demonstrates various ways to configure DocuChat using the dual
configuration system (YAML files + command-line arguments).
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import docuchat
sys.path.append(str(Path(__file__).parent.parent))

from docuchat import DocuChatConfig


def example_yaml_only():
    """
    Example 1: Using YAML configuration only
    
    This loads all settings from a YAML file without any command-line overrides.
    """
    print("=== Example 1: YAML Configuration Only ===")
    
    # Load configuration from YAML file
    config = DocuChatConfig.from_yaml("config/config.yaml")
    
    print(f"Model path: {config.model_path}")
    print(f"Folder path: {config.folder_path}")
    print(f"Chunk size: {config.chunk_size}")
    print(f"Temperature: {config.temperature}")
    print(f"Embedding model: {config.embedding_model}")
    print()


def example_command_line_only():
    """
    Example 2: Command-line arguments only
    
    This demonstrates using DocuChat without any YAML configuration,
    relying entirely on command-line arguments or defaults.
    """
    print("=== Example 2: Command-line Only ===")
    
    # Create default configuration
    config = DocuChatConfig()
    
    # Simulate command-line arguments
    class MockArgs:
        def __init__(self):
            self.model_path = "./models/llama-2-7b.gguf"
            self.folder_path = "./documents"
            self.chunk_size = 1500
            self.chunk_overlap = 300
            self.embedding_model = "all-mpnet-base-v2"
            self.n_ctx = 8192
            self.temperature = 0.3
            self.max_tokens = 4096
            self.n_retrieve = 8
            self.verbose = True
            self.system_prompt = "You are an expert research assistant."
            self.chat_template = "chatml"
    
    args = MockArgs()
    config.update_from_args(args)
    
    print(f"Model path: {config.model_path}")
    print(f"Folder path: {config.folder_path}")
    print(f"Chunk size: {config.chunk_size}")
    print(f"Temperature: {config.temperature}")
    print(f"Context length: {config.n_ctx}")
    print(f"Verbose: {config.verbose}")
    print()


def example_mixed_configuration():
    """
    Example 3: Mixed configuration (YAML + command-line overrides)
    
    This shows how to load base settings from YAML and override specific
    settings with command-line arguments.
    """
    print("=== Example 3: Mixed Configuration ===")
    
    # Load base configuration from YAML
    config = DocuChatConfig.from_yaml("config/config.yaml")
    
    print("Before command-line overrides:")
    print(f"  Model path: {config.model_path}")
    print(f"  Temperature: {config.temperature}")
    print(f"  Chunk size: {config.chunk_size}")
    
    # Simulate command-line overrides
    class MockArgs:
        def __init__(self):
            self.model_path = "./models/custom-model.gguf"  # Override
            self.folder_path = None  # Don't override
            self.chunk_size = 1000  # Default value, won't override
            self.chunk_overlap = 200  # Default value, won't override
            self.embedding_model = "all-MiniLM-L6-v2"  # Default, won't override
            self.n_ctx = 4096  # Default value, won't override
            self.temperature = 0.9  # Override
            self.max_tokens = 2048  # Default value, won't override
            self.n_retrieve = 5  # Default value, won't override
            self.verbose = True  # Override
            self.system_prompt = None  # Don't override
            self.chat_template = "auto"  # Default, won't override
    
    args = MockArgs()
    config.update_from_args(args)
    
    print("\nAfter command-line overrides:")
    print(f"  Model path: {config.model_path}")
    print(f"  Temperature: {config.temperature}")
    print(f"  Chunk size: {config.chunk_size}")
    print(f"  Verbose: {config.verbose}")
    print()


def explain_configuration_priority():
    """
    Explanation of configuration priority and best practices.
    """
    print("=== Configuration Priority & Best Practices ===")
    print("""
Configuration Priority (later overrides earlier):
1. Default values in DocuChatConfig class
2. YAML configuration file settings
3. Command-line argument overrides

Best Practices:

1. YAML for Base Configuration:
   - Use YAML files for common settings across runs
   - Store different configs for different scenarios (dev, prod, etc.)
   - Keep sensitive settings out of version control

2. Command-line for Overrides:
   - Override specific settings for testing
   - Provide different model paths or document folders
   - Adjust parameters for specific runs

3. Environment-specific Configs:
   - config/development.yaml
   - config/production.yaml
   - config/testing.yaml

4. Security Considerations:
   - Don't commit API keys or sensitive paths to version control
   - Use environment variables for sensitive settings
   - Consider using .env files for local development
    """)
    print()


def show_yaml_structure():
    """
    Show the structure of a typical YAML configuration file.
    """
    print("=== YAML Configuration Structure ===")
    print("""
Typical config.yaml structure:

model:
  path: "./models/llama-2-7b.gguf"
  context_length: 4096
  temperature: 0.7
  max_tokens: 2048

documents:
  folder_path: "./documents"
  chunk_size: 1000
  chunk_overlap: 200

embeddings:
  model: "all-MiniLM-L6-v2"

vector_store:
  collection_name: "documents"

rag:
  retrieve_count: 5
  similarity_threshold: 0.7

ui:
  system_prompt: "You are a helpful assistant..."
  chat_template: "auto"
  verbose: false

performance:
  batch_size: 512
  use_mlock: false
  use_mmap: true
    """)
    print()


def show_common_workflows():
    """
    Show common configuration workflows.
    """
    print("=== Common Configuration Workflows ===")
    print("""
1. Development Workflow:
   # Use development config with verbose logging
   python docuchat.py --config config/development.yaml --verbose

2. Production Workflow:
   # Use production config with optimized settings
   python docuchat.py --config config/production.yaml

3. Testing Different Models:
   # Keep base config, test different models
   python docuchat.py --model_path ./models/model1.gguf
   python docuchat.py --model_path ./models/model2.gguf

4. Document-specific Runs:
   # Different document sets with same model
   python docuchat.py --folder_path ./legal_docs
   python docuchat.py --folder_path ./technical_docs

5. Parameter Tuning:
   # Test different RAG parameters
   python docuchat.py --n_retrieve 3 --temperature 0.3
   python docuchat.py --n_retrieve 10 --temperature 0.9

6. Single Query Testing:
   # Test specific queries without interactive mode
   python docuchat.py --query "What is the main topic?"
    """)
    print()


def main():
    """
    Run all configuration examples.
    """
    print("DocuChat Configuration Examples\n")
    print("=" * 50)
    
    # Run examples
    example_yaml_only()
    example_command_line_only()
    example_mixed_configuration()
    
    # Show explanations
    explain_configuration_priority()
    show_yaml_structure()
    show_common_workflows()
    
    print("=" * 50)
    print("\nFor more information, see the README.md file.")


if __name__ == "__main__":
    main()
