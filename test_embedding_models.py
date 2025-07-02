#!/usr/bin/env python3
"""
Test script to verify custom embedding model functionality.
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

from docuchat import VectorStore

def test_embedding_models():
    """Test different embedding model configurations."""
    
    print("🧪 Testing Embedding Model Functionality\n")
    
    # Test cases
    test_cases = [
        {
            "name": "Default Model",
            "model": "all-MiniLM-L6-v2",
            "expected_behavior": "Should use Hugging Face model"
        },
        {
            "name": "Alternative HF Model", 
            "model": "all-mpnet-base-v2",
            "expected_behavior": "Should download and use MPNet model"
        },
        {
            "name": "Non-existent Local Model",
            "model": "non-existent-model",
            "expected_behavior": "Should treat as HF identifier"
        },
        {
            "name": "Relative Path (non-existent)",
            "model": "./embeddings/custom-model",
            "expected_behavior": "Should use path as-is"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['name']}")
        print(f"Model: {test_case['model']}")
        print(f"Expected: {test_case['expected_behavior']}")
        
        try:
            # Create VectorStore instance
            vs = VectorStore(
                collection_name=f"test_{i}",
                embedding_model=test_case['model']
            )
            
            # Test model resolution
            resolved_path = vs._resolve_embedding_model_path(test_case['model'])
            print(f"✅ Resolved path: {resolved_path}")
            print(f"✅ Model loaded successfully: {vs.embedding_model_name}")
            
            # Test embedding generation
            test_docs = ["This is a test document.", "Another test document."]
            embeddings = vs.embedding_model.encode(test_docs)
            print(f"✅ Generated embeddings shape: {embeddings.shape}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 50)
    
    print("\n🎯 Testing Complete!")
    print("\n📁 Embeddings folder structure:")
    embeddings_dir = Path("embeddings")
    if embeddings_dir.exists():
        for item in embeddings_dir.iterdir():
            print(f"  - {item.name}")
    else:
        print("  - Embeddings folder not found")

if __name__ == "__main__":
    test_embedding_models()