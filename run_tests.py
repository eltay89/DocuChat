#!/usr/bin/env python3
"""
DocuChat v2.0 Test Runner

Simple test runner script for DocuChat tests.
This script provides an easy way to run tests without requiring pytest installation.
"""

import sys
import os
import subprocess
from pathlib import Path

def check_pytest_installed():
    """Check if pytest is installed."""
    try:
        import pytest
        return True
    except ImportError:
        return False

def install_pytest():
    """Install pytest if not available."""
    print("Installing pytest...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pytest", "pytest-cov"])
        print("✅ pytest installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install pytest")
        return False

def run_tests(test_file=None, verbose=False, coverage=False):
    """Run tests with optional parameters."""
    if not check_pytest_installed():
        if not install_pytest():
            print("Cannot run tests without pytest. Please install it manually:")
            print("pip install pytest pytest-cov")
            return False
    
    # Build pytest command
    cmd = [sys.executable, "-m", "pytest"]
    
    if test_file:
        cmd.append(f"tests/{test_file}")
    else:
        cmd.append("tests/")
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=src/", "--cov-report=html", "--cov-report=term"])
    
    # Add current directory to Python path
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path.cwd())
    
    print(f"Running command: {' '.join(cmd)}")
    print("=" * 50)
    
    try:
        result = subprocess.run(cmd, env=env)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running tests: {e}")
        return False

def main():
    """Main test runner function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="DocuChat v2.0 Test Runner")
    parser.add_argument(
        "--file", 
        help="Specific test file to run (e.g., test_tools.py)"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Verbose output"
    )
    parser.add_argument(
        "--coverage", "-c", 
        action="store_true", 
        help="Run with coverage report"
    )
    parser.add_argument(
        "--list", "-l", 
        action="store_true", 
        help="List available test files"
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("Available test files:")
        test_dir = Path("tests")
        if test_dir.exists():
            for test_file in test_dir.glob("test_*.py"):
                print(f"  - {test_file.name}")
        else:
            print("  No tests directory found")
        return
    
    print("🧪 DocuChat v2.0 Test Runner")
    print("=" * 30)
    
    # Check if tests directory exists
    if not Path("tests").exists():
        print("❌ Tests directory not found")
        return
    
    # Run tests
    success = run_tests(
        test_file=args.file,
        verbose=args.verbose,
        coverage=args.coverage
    )
    
    if success:
        print("\n✅ All tests passed!")
        if args.coverage:
            print("📊 Coverage report generated in htmlcov/")
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
