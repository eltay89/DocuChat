#!/bin/bash

# DocuChat v2.0 Docker Setup Script
# This script helps set up DocuChat using Docker

set -e  # Exit on any error

echo "🐳 DocuChat v2.0 Docker Setup"
echo "============================="

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

# Check if Docker is installed
check_docker() {
    print_status "Checking Docker installation..."
    if command -v docker &> /dev/null; then
        DOCKER_VERSION=$(docker --version | cut -d" " -f3 | cut -d"," -f1)
        print_success "Docker $DOCKER_VERSION found"
    else
        print_error "Docker is not installed. Please install Docker first."
        echo "Visit: https://docs.docker.com/get-docker/"
        exit 1
    fi
}

# Check if Docker Compose is installed
check_docker_compose() {
    print_status "Checking Docker Compose installation..."
    if command -v docker-compose &> /dev/null; then
        COMPOSE_VERSION=$(docker-compose --version | cut -d" " -f3 | cut -d"," -f1)
        print_success "Docker Compose $COMPOSE_VERSION found"
    else
        print_error "Docker Compose is not installed. Please install Docker Compose."
        echo "Visit: https://docs.docker.com/compose/install/"
        exit 1
    fi
}

# Check if Docker daemon is running
check_docker_daemon() {
    print_status "Checking Docker daemon..."
    if docker info &> /dev/null; then
        print_success "Docker daemon is running"
    else
        print_error "Docker daemon is not running. Please start Docker."
        exit 1
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

# Setup configuration
setup_config() {
    print_status "Setting up configuration..."
    if [ ! -f "config/config.yaml" ]; then
        if [ -f "config/config.yaml.template" ]; then
            cp config/config.yaml.template config/config.yaml
            print_success "Configuration template copied"
            print_warning "Please edit config/config.yaml with your API keys"
        else
            print_warning "Configuration template not found, using defaults"
        fi
    else
        print_warning "Configuration file already exists"
    fi
}

# Build Docker images
build_images() {
    print_status "Building Docker images..."
    docker-compose build
    print_success "Docker images built successfully"
}

# Start services
start_services() {
    print_status "Starting DocuChat services..."
    docker-compose up -d
    print_success "Services started successfully"
}

# Check service status
check_services() {
    print_status "Checking service status..."
    docker-compose ps
    
    # Wait for services to be ready
    sleep 5
    
    # Check if web service is responding
    if curl -f http://localhost:8000/health &> /dev/null; then
        print_success "Web service is responding"
    else
        print_warning "Web service may not be ready yet"
    fi
    
    # Check if Streamlit service is responding
    if curl -f http://localhost:8501 &> /dev/null; then
        print_success "Streamlit service is responding"
    else
        print_warning "Streamlit service may not be ready yet"
    fi
}

# Show logs
show_logs() {
    if [ "$1" = "--logs" ] || [ "$1" = "-l" ]; then
        print_status "Showing service logs..."
        docker-compose logs --tail=50
    fi
}

# Stop services
stop_services() {
    if [ "$1" = "--stop" ] || [ "$1" = "-s" ]; then
        print_status "Stopping DocuChat services..."
        docker-compose down
        print_success "Services stopped"
        exit 0
    fi
}

# Clean up
cleanup() {
    if [ "$1" = "--clean" ] || [ "$1" = "-c" ]; then
        print_status "Cleaning up Docker resources..."
        docker-compose down -v --rmi all
        docker system prune -f
        print_success "Cleanup completed"
        exit 0
    fi
}

# Print usage information
print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --logs, -l        Show service logs after startup"
    echo "  --stop, -s        Stop running services"
    echo "  --clean, -c       Clean up all Docker resources"
    echo "  --help, -h        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                Start DocuChat services"
    echo "  $0 --logs         Start services and show logs"
    echo "  $0 --stop         Stop running services"
    echo "  $0 --clean        Clean up all resources"
}

# Main setup function
main() {
    # Parse command line arguments
    for arg in "$@"; do
        case $arg in
            --stop|-s)
                stop_services --stop
                ;;
            --clean|-c)
                cleanup --clean
                ;;
            --help|-h)
                print_usage
                exit 0
                ;;
        esac
    done
    
    # Run setup steps
    check_docker
    check_docker_compose
    check_docker_daemon
    create_directories
    setup_config
    build_images
    start_services
    check_services
    
    # Show logs if requested
    show_logs "$1"
    
    # Print completion message
    echo ""
    print_success "🎉 DocuChat v2.0 is now running!"
    echo ""
    print_status "Available interfaces:"
    echo "• CLI: docker-compose exec cli python -m docuchat.cli.main"
    echo "• Web: http://localhost:8000"
    echo "• Streamlit: http://localhost:8501"
    echo ""
    print_status "Useful commands:"
    echo "• View logs: docker-compose logs -f"
    echo "• Stop services: docker-compose down"
    echo "• Restart: docker-compose restart"
    echo ""
    print_status "For more information, see README.md"
}

# Run main function with all arguments
main "$@"
