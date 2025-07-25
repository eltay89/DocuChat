#!/bin/bash

# DocuChat v2.0 Setup Script
# This script helps set up DocuChat for development or production use

set -e  # Exit on any error

echo "🚀 DocuChat v2.0 Setup Script"
echo "=============================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is installed
check_python() {
    print_status "Checking Python installation..."
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d" " -f2)
        print_success "Python $PYTHON_VERSION found"
        
        # Check if version is 3.8 or higher
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
            print_success "Python version is compatible (3.8+)"
        else
            print_error "Python 3.8 or higher is required. Current version: $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 is not installed. Please install Python 3.8 or higher."
        exit 1
    fi
}

# Check if pip is installed
check_pip() {
    print_status "Checking pip installation..."
    if command -v pip3 &> /dev/null; then
        print_success "pip3 found"
    else
        print_error "pip3 is not installed. Please install pip."
        exit 1
    fi
}

# Create virtual environment
create_venv() {
    print_status "Creating virtual environment..."
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
}

# Activate virtual environment
activate_venv() {
    print_status "Activating virtual environment..."
    source venv/bin/activate
    print_success "Virtual environment activated"
}

# Install dependencies
install_dependencies() {
    print_status "Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    print_success "Dependencies installed"
}

# Install enhanced dependencies
install_enhanced() {
    if [ "$1" = "--enhanced" ] || [ "$1" = "-e" ]; then
        print_status "Installing enhanced dependencies..."
        pip install "unstructured[all-docs]>=0.10.0"
        pip install "easyocr>=1.7.0"
        pip install "sentence-transformers>=2.2.0"
        pip install "rank-bm25>=0.2.2"
        print_success "Enhanced dependencies installed"
    fi
}

# Install web dependencies
install_web() {
    if [ "$1" = "--web" ] || [ "$1" = "-w" ]; then
        print_status "Installing web dependencies..."
        pip install "fastapi>=0.100.0"
        pip install "uvicorn[standard]>=0.23.0"
        pip install "streamlit>=1.25.0"
        print_success "Web dependencies installed"
    fi
}

# Install development dependencies
install_dev() {
    if [ "$1" = "--dev" ] || [ "$1" = "-d" ]; then
        print_status "Installing development dependencies..."
        pip install "pytest>=7.0.0"
        pip install "pytest-cov>=4.0.0"
        pip install "black>=23.0.0"
        pip install "isort>=5.12.0"
        pip install "flake8>=6.0.0"
        pip install "mypy>=1.5.0"
        pip install "pre-commit>=3.0.0"
        print_success "Development dependencies installed"
    fi
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    mkdir -p documents
    mkdir -p vector_store
    mkdir -p logs
    mkdir -p config
    print_success "Directories created"
}

# Copy configuration template
setup_config() {
    print_status "Setting up configuration..."
    if [ ! -f "config/config.yaml" ]; then
        if [ -f "config/config.yaml.template" ]; then
            cp config/config.yaml.template config/config.yaml
            print_success "Configuration template copied to config/config.yaml"
            print_warning "Please edit config/config.yaml with your settings"
        else
            print_error "Configuration template not found"
        fi
    else
        print_warning "Configuration file already exists"
    fi
}

# Install DocuChat package
install_package() {
    print_status "Installing DocuChat package..."
    pip install -e .
    print_success "DocuChat package installed in development mode"
}

# Test installation
test_installation() {
    print_status "Testing installation..."
    if python3 -c "import docuchat; print('DocuChat imported successfully')"; then
        print_success "Installation test passed"
    else
        print_error "Installation test failed"
        exit 1
    fi
}

# Setup pre-commit hooks
setup_precommit() {
    if [ "$1" = "--dev" ] || [ "$1" = "-d" ]; then
        print_status "Setting up pre-commit hooks..."
        pre-commit install
        print_success "Pre-commit hooks installed"
    fi
}

# Print usage information
print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --enhanced, -e    Install enhanced features (OCR, advanced processing)"
    echo "  --web, -w         Install web interface dependencies"
    echo "  --dev, -d         Install development dependencies"
    echo "  --help, -h        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                Basic installation"
    echo "  $0 --enhanced     Install with enhanced features"
    echo "  $0 --web --dev    Install with web and development features"
}

# Main setup function
main() {
    # Parse command line arguments
    ENHANCED=false
    WEB=false
    DEV=false
    
    for arg in "$@"; do
        case $arg in
            --enhanced|-e)
                ENHANCED=true
                ;;
            --web|-w)
                WEB=true
                ;;
            --dev|-d)
                DEV=true
                ;;
            --help|-h)
                print_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $arg"
                print_usage
                exit 1
                ;;
        esac
    done
    
    # Run setup steps
    check_python
    check_pip
    create_venv
    activate_venv
    install_dependencies
    
    # Install optional dependencies
    if [ "$ENHANCED" = true ]; then
        install_enhanced --enhanced
    fi
    
    if [ "$WEB" = true ]; then
        install_web --web
    fi
    
    if [ "$DEV" = true ]; then
        install_dev --dev
    fi
    
    create_directories
    setup_config
    install_package
    test_installation
    
    if [ "$DEV" = true ]; then
        setup_precommit --dev
    fi
    
    # Print completion message
    echo ""
    print_success "🎉 DocuChat v2.0 setup completed successfully!"
    echo ""
    print_status "Next steps:"
    echo "1. Edit config/config.yaml with your API keys and preferences"
    echo "2. Add documents to the documents/ directory"
    echo "3. Run DocuChat with: python -m docuchat.cli.main"
    echo ""
    print_status "For more information, see README.md"
}

# Run main function with all arguments
main "$@"
