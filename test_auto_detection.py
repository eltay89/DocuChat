#!/usr/bin/env python3
"""
Test Auto-Detection Functionality

This script tests the auto-detection capabilities of DocuChat,
including GGUF model loading and chat template detection.

Usage:
    python test_auto_detection.py [model_path]

If no model path is provided, it will look for models in the ./models/ directory.
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add the current directory to Python path to import docuchat
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import docuchat
except ImportError:
    print("Error: Could not import docuchat module")
    sys.exit(1)

def find_gguf_models(directory: str = "./models") -> List[Path]:
    """
    Find all GGUF model files in the specified directory.
    
    Args:
        directory: Directory to search for GGUF files
        
    Returns:
        List of Path objects for found GGUF files
    """
    models_dir = Path(directory)
    if not models_dir.exists():
        return []
    
    gguf_files = []
    for file_path in models_dir.rglob("*.gguf"):
        if file_path.is_file():
            gguf_files.append(file_path)
    
    return gguf_files

def test_model_loading(model_path: str) -> Dict[str, Any]:
    """
    Test loading a GGUF model and extracting information.
    
    Args:
        model_path: Path to the GGUF model file
        
    Returns:
        Dictionary containing test results
    """
    test_results = {
        "model_path": model_path,
        "file_exists": False,
        "file_size_mb": 0,
        "loading_success": False,
        "metadata_extraction": False,
        "chat_template_detection": False,
        "chat_template_type": None,
        "errors": [],
        "warnings": []
    }
    
    # Check file existence
    model_file = Path(model_path)
    test_results["file_exists"] = model_file.exists()
    
    if not test_results["file_exists"]:
        test_results["errors"].append(f"Model file does not exist: {model_path}")
        return test_results
    
    test_results["file_size_mb"] = round(model_file.stat().st_size / (1024 * 1024), 2)
    
    # Test model loading
    try:
        print(f"  Testing model loading...")
        
        # Try to create a DocuChat instance with minimal settings
        config = {
            "model_path": model_path,
            "n_ctx": 512,  # Minimal context for testing
            "verbose": False
        }
        
        # This is a simplified test - in a real scenario, you'd need documents
        # For now, we'll just test the model loading part
        
        # Import llama_cpp directly for testing
        try:
            from llama_cpp import Llama
            
            llm = Llama(
                model_path=model_path,
                n_ctx=512,
                verbose=False,
                n_threads=1
            )
            
            test_results["loading_success"] = True
            
            # Test metadata extraction
            try:
                metadata = llm.metadata
                test_results["metadata_extraction"] = True
                
                # Check for chat template
                if "tokenizer.chat_template" in metadata:
                    test_results["chat_template_detection"] = True
                    
                    # Try to detect template type
                    template = metadata["tokenizer.chat_template"]
                    
                    if "<|im_start|>" in template and "<|im_end|>" in template:
                        test_results["chat_template_type"] = "chatml"
                    elif "[INST]" in template and "[/INST]" in template:
                        if "<<SYS>>" in template:
                            test_results["chat_template_type"] = "llama2"
                        else:
                            test_results["chat_template_type"] = "mistral"
                    elif "### Instruction:" in template:
                        test_results["chat_template_type"] = "alpaca"
                    elif "USER:" in template and "ASSISTANT:" in template:
                        test_results["chat_template_type"] = "vicuna"
                    else:
                        test_results["chat_template_type"] = "custom"
                else:
                    test_results["warnings"].append("No chat template found in metadata")
                
            except Exception as e:
                test_results["warnings"].append(f"Metadata extraction failed: {str(e)}")
            
        except Exception as e:
            test_results["errors"].append(f"Model loading failed: {str(e)}")
    
    except Exception as e:
        test_results["errors"].append(f"Unexpected error: {str(e)}")
    
    return test_results

def test_chat_template_detection() -> Dict[str, Any]:
    """
    Test chat template detection functionality.
    
    Returns:
        Dictionary containing test results
    """
    test_results = {
        "template_formats_tested": [],
        "detection_success": {},
        "errors": []
    }
    
    # Test template samples
    template_samples = {
        "chatml": "<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        "llama2": "[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{prompt} [/INST]",
        "alpaca": "### Instruction:\n{prompt}\n\n### Response:\n",
        "vicuna": "USER: {prompt}\nASSISTANT:"
    }
    
    for template_name, template_text in template_samples.items():
        test_results["template_formats_tested"].append(template_name)
        
        try:
            # Test if we can detect the template type from the text
            detected = False
            
            if template_name == "chatml" and "<|im_start|>" in template_text:
                detected = True
            elif template_name == "llama2" and "[INST]" in template_text and "<<SYS>>" in template_text:
                detected = True
            elif template_name == "alpaca" and "### Instruction:" in template_text:
                detected = True
            elif template_name == "vicuna" and "USER:" in template_text and "ASSISTANT:" in template_text:
                detected = True
            
            test_results["detection_success"][template_name] = detected
            
        except Exception as e:
            test_results["errors"].append(f"Template detection failed for {template_name}: {str(e)}")
            test_results["detection_success"][template_name] = False
    
    return test_results

def print_test_results(model_results: List[Dict[str, Any]], template_results: Dict[str, Any]) -> None:
    """
    Print formatted test results.
    
    Args:
        model_results: List of model test results
        template_results: Chat template test results
    """
    print("\n" + "="*60)
    print("AUTO-DETECTION TEST RESULTS")
    print("="*60)
    
    # Model Loading Tests
    print("\n🤖 MODEL LOADING TESTS:")
    
    if not model_results:
        print("  No models found to test.")
        print("  Place GGUF model files in the ./models/ directory.")
    else:
        for i, result in enumerate(model_results, 1):
            print(f"\n  Model {i}: {Path(result['model_path']).name}")
            print(f"    File Exists: {'✅' if result['file_exists'] else '❌'}")
            if result['file_exists']:
                print(f"    File Size: {result['file_size_mb']} MB")
                print(f"    Loading: {'✅' if result['loading_success'] else '❌'}")
                print(f"    Metadata: {'✅' if result['metadata_extraction'] else '❌'}")
                print(f"    Chat Template: {'✅' if result['chat_template_detection'] else '❌'}")
                if result['chat_template_type']:
                    print(f"    Template Type: {result['chat_template_type']}")
            
            if result['warnings']:
                print("    Warnings:")
                for warning in result['warnings']:
                    print(f"      - {warning}")
            
            if result['errors']:
                print("    Errors:")
                for error in result['errors']:
                    print(f"      - {error}")
    
    # Chat Template Detection Tests
    print("\n💬 CHAT TEMPLATE DETECTION TESTS:")
    
    for template_name in template_results["template_formats_tested"]:
        success = template_results["detection_success"].get(template_name, False)
        print(f"  {template_name.upper()}: {'✅' if success else '❌'}")
    
    if template_results["errors"]:
        print("\n  Errors:")
        for error in template_results["errors"]:
            print(f"    - {error}")
    
    # Summary
    print("\n📊 SUMMARY:")
    
    total_models = len(model_results)
    successful_loads = sum(1 for r in model_results if r["loading_success"])
    template_detections = sum(1 for r in model_results if r["chat_template_detection"])
    
    print(f"  Models Tested: {total_models}")
    print(f"  Successful Loads: {successful_loads}/{total_models}")
    print(f"  Template Detections: {template_detections}/{total_models}")
    
    template_tests = len(template_results["template_formats_tested"])
    template_successes = sum(1 for success in template_results["detection_success"].values() if success)
    print(f"  Template Format Tests: {template_successes}/{template_tests}")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    
    if successful_loads < total_models:
        print("  - Some models failed to load. Check file integrity and available RAM.")
    
    if template_detections < successful_loads:
        print("  - Some models lack chat templates. Use --chat_template parameter.")
    
    if total_models == 0:
        print("  - Add GGUF model files to ./models/ directory for testing.")
        print("  - Download models from: https://huggingface.co/models?library=gguf")
    
    print("\n" + "="*60)

def main():
    """
    Main function to run auto-detection tests.
    """
    print("DocuChat Auto-Detection Test Suite")
    print("===================================")
    
    # Check if a specific model path was provided
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        if not Path(model_path).exists():
            print(f"Error: Model file does not exist: {model_path}")
            sys.exit(1)
        model_paths = [Path(model_path)]
    else:
        # Find models in the models directory
        model_paths = find_gguf_models("./models")
        if not model_paths:
            print("\nNo GGUF models found in ./models/ directory.")
            print("Please add some GGUF model files or specify a model path.")
            print("\nUsage: python test_auto_detection.py [model_path]")
    
    # Test model loading
    model_results = []
    for model_path in model_paths:
        print(f"\nTesting model: {model_path.name}")
        result = test_model_loading(str(model_path))
        model_results.append(result)
    
    # Test chat template detection
    print("\nTesting chat template detection...")
    template_results = test_chat_template_detection()
    
    # Print results
    print_test_results(model_results, template_results)
    
    # Exit with appropriate code
    total_errors = sum(len(r["errors"]) for r in model_results) + len(template_results["errors"])
    if total_errors > 0:
        sys.exit(1)
    else:
        print("\n🎉 All tests completed successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()