#!/usr/bin/env python3
"""
GGUF Debug Utility

This script helps debug GGUF model loading issues by providing detailed
information about model metadata, chat templates, and potential problems.

Usage:
    python debug_gguf.py <path_to_gguf_model>

Example:
    python debug_gguf.py ./models/llama-2-7b-chat.Q4_K_M.gguf
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from llama_cpp import Llama
except ImportError:
    print("Error: llama-cpp-python not installed. Install with: pip install llama-cpp-python")
    sys.exit(1)

def debug_gguf_model(model_path: str) -> Dict[str, Any]:
    """
    Debug a GGUF model and extract detailed information.
    
    Args:
        model_path: Path to the GGUF model file
        
    Returns:
        Dictionary containing debug information
    """
    debug_info = {
        "file_info": {},
        "model_info": {},
        "chat_template": {},
        "errors": [],
        "warnings": []
    }
    
    # Check file existence and basic info
    model_file = Path(model_path)
    if not model_file.exists():
        debug_info["errors"].append(f"Model file does not exist: {model_path}")
        return debug_info
    
    debug_info["file_info"] = {
        "path": str(model_file.absolute()),
        "size_mb": round(model_file.stat().st_size / (1024 * 1024), 2),
        "exists": True,
        "readable": os.access(model_file, os.R_OK)
    }
    
    if not debug_info["file_info"]["readable"]:
        debug_info["errors"].append("Model file is not readable")
        return debug_info
    
    # Try to load the model with minimal settings
    try:
        print(f"Loading model: {model_path}")
        print("This may take a moment...")
        
        # Load with minimal context to speed up loading
        llm = Llama(
            model_path=model_path,
            n_ctx=512,  # Minimal context for testing
            verbose=False,
            n_threads=1
        )
        
        debug_info["model_info"]["loaded_successfully"] = True
        
        # Get model metadata
        try:
            metadata = llm.metadata
            debug_info["model_info"]["metadata"] = metadata
            
            # Extract common metadata fields
            common_fields = [
                "general.name",
                "general.architecture", 
                "general.quantization_version",
                "general.file_type",
                "llama.context_length",
                "llama.embedding_length",
                "llama.block_count",
                "llama.feed_forward_length",
                "llama.attention.head_count",
                "tokenizer.ggml.model",
                "tokenizer.chat_template"
            ]
            
            extracted_metadata = {}
            for field in common_fields:
                if field in metadata:
                    extracted_metadata[field] = metadata[field]
            
            debug_info["model_info"]["key_metadata"] = extracted_metadata
            
        except Exception as e:
            debug_info["warnings"].append(f"Could not extract metadata: {str(e)}")
        
        # Check for chat template
        chat_template_info = {}
        
        # Method 1: Check metadata
        if "tokenizer.chat_template" in debug_info["model_info"].get("metadata", {}):
            template = debug_info["model_info"]["metadata"]["tokenizer.chat_template"]
            chat_template_info["from_metadata"] = {
                "found": True,
                "template": template[:200] + "..." if len(template) > 200 else template,
                "full_length": len(template)
            }
        else:
            chat_template_info["from_metadata"] = {"found": False}
        
        # Method 2: Try to detect template type
        template_indicators = {
            "chatml": ["<|im_start|>", "<|im_end|>"],
            "llama2": ["[INST]", "[/INST]", "<<SYS>>", "<</SYS>>"],
            "alpaca": ["### Instruction:", "### Response:"],
            "vicuna": ["USER:", "ASSISTANT:"],
            "mistral": ["[INST]", "[/INST]"] # Similar to Llama2 but different context
        }
        
        detected_templates = []
        if "from_metadata" in chat_template_info and chat_template_info["from_metadata"]["found"]:
            template_text = debug_info["model_info"]["metadata"]["tokenizer.chat_template"]
            for template_name, indicators in template_indicators.items():
                if any(indicator in template_text for indicator in indicators):
                    detected_templates.append(template_name)
        
        chat_template_info["detected_types"] = detected_templates
        
        # Method 3: Test basic generation
        try:
            test_prompt = "Hello"
            response = llm(test_prompt, max_tokens=10, echo=False)
            chat_template_info["generation_test"] = {
                "success": True,
                "prompt": test_prompt,
                "response": response["choices"][0]["text"] if response.get("choices") else "No response"
            }
        except Exception as e:
            chat_template_info["generation_test"] = {
                "success": False,
                "error": str(e)
            }
        
        debug_info["chat_template"] = chat_template_info
        
    except Exception as e:
        debug_info["model_info"]["loaded_successfully"] = False
        debug_info["errors"].append(f"Failed to load model: {str(e)}")
    
    return debug_info

def print_debug_report(debug_info: Dict[str, Any]) -> None:
    """
    Print a formatted debug report.
    
    Args:
        debug_info: Debug information dictionary
    """
    print("\n" + "="*60)
    print("GGUF MODEL DEBUG REPORT")
    print("="*60)
    
    # File Information
    print("\n📁 FILE INFORMATION:")
    file_info = debug_info["file_info"]
    if file_info:
        print(f"  Path: {file_info.get('path', 'Unknown')}")
        print(f"  Size: {file_info.get('size_mb', 0)} MB")
        print(f"  Exists: {file_info.get('exists', False)}")
        print(f"  Readable: {file_info.get('readable', False)}")
    
    # Model Information
    print("\n🤖 MODEL INFORMATION:")
    model_info = debug_info["model_info"]
    print(f"  Loaded Successfully: {model_info.get('loaded_successfully', False)}")
    
    if "key_metadata" in model_info:
        print("\n  Key Metadata:")
        for key, value in model_info["key_metadata"].items():
            print(f"    {key}: {value}")
    
    # Chat Template Information
    print("\n💬 CHAT TEMPLATE INFORMATION:")
    chat_info = debug_info["chat_template"]
    
    if "from_metadata" in chat_info:
        metadata_info = chat_info["from_metadata"]
        print(f"  Found in Metadata: {metadata_info.get('found', False)}")
        if metadata_info.get("found"):
            print(f"  Template Length: {metadata_info.get('full_length', 0)} characters")
            print(f"  Template Preview: {metadata_info.get('template', 'N/A')}")
    
    if "detected_types" in chat_info:
        detected = chat_info["detected_types"]
        print(f"  Detected Template Types: {', '.join(detected) if detected else 'None'}")
    
    if "generation_test" in chat_info:
        gen_test = chat_info["generation_test"]
        print(f"  Generation Test: {'✅ Success' if gen_test.get('success') else '❌ Failed'}")
        if gen_test.get("success"):
            print(f"    Test Response: {gen_test.get('response', 'N/A')}")
        else:
            print(f"    Error: {gen_test.get('error', 'Unknown')}")
    
    # Warnings
    if debug_info["warnings"]:
        print("\n⚠️  WARNINGS:")
        for warning in debug_info["warnings"]:
            print(f"  - {warning}")
    
    # Errors
    if debug_info["errors"]:
        print("\n❌ ERRORS:")
        for error in debug_info["errors"]:
            print(f"  - {error}")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    
    if not debug_info["model_info"].get("loaded_successfully", False):
        print("  - Model failed to load. Check file path and format.")
        print("  - Ensure you have enough RAM for the model.")
        print("  - Try a smaller model or reduce n_ctx parameter.")
    
    if not debug_info["chat_template"].get("from_metadata", {}).get("found", False):
        print("  - No chat template found in metadata.")
        print("  - Use --chat_template parameter to specify format manually.")
        print("  - Try: chatml, llama2, alpaca, or vicuna")
    
    if debug_info["chat_template"].get("detected_types"):
        detected = debug_info["chat_template"]["detected_types"]
        print(f"  - Detected template types: {', '.join(detected)}")
        print(f"  - Try using: --chat_template {detected[0]}")
    
    print("\n" + "="*60)

def main():
    """
    Main function to run the debug utility.
    """
    if len(sys.argv) != 2:
        print("Usage: python debug_gguf.py <path_to_gguf_model>")
        print("\nExample:")
        print("  python debug_gguf.py ./models/llama-2-7b-chat.Q4_K_M.gguf")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    print(f"Debugging GGUF model: {model_path}")
    print("This may take a moment to load the model...")
    
    try:
        debug_info = debug_gguf_model(model_path)
        print_debug_report(debug_info)
        
        # Exit with appropriate code
        if debug_info["errors"]:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n\nDebug interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error during debugging: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()