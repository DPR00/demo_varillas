#!/bin/bash

# Demo Varillas - Master Runner Script
# This script provides a menu to run all different implementations

set -e  # Exit on any error

# Get the project root directory (two levels up from docker/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Function to check if Docker images exist
check_docker_images() {
    echo "🔍 Checking Docker images..."
    
    # Check for demo-varillas image
    if docker images --format "table {{.Repository}}:{{.Tag}}" | grep -q "demo-varillas.*latest"; then
        print_status "demo-varillas image found"
        IMAGE_EXISTS=true
    else
        print_warning "demo-varillas image not found"
        IMAGE_EXISTS=false
    fi
}

# Function to check prerequisites
check_prerequisites() {
    echo "🔍 Checking prerequisites..."
    
    # Check if Docker is available
    if command -v docker &> /dev/null; then
        if docker info > /dev/null 2>&1; then
            print_status "Docker is running"
            DOCKER_AVAILABLE=true
        else
            print_warning "Docker is installed but not running"
            DOCKER_AVAILABLE=false
        fi
    else
        print_warning "Docker is not installed"
        DOCKER_AVAILABLE=false
    fi
    
    # Check if docker compose is available
    if docker compose version > /dev/null 2>&1; then
        print_status "docker compose is available"
        COMPOSE_AVAILABLE=true
    else
        print_warning "docker compose is not available"
        COMPOSE_AVAILABLE=false
    fi
    
    # Check if Python is available
    if command -v python3 &> /dev/null; then
        print_status "Python 3 is available"
        PYTHON_AVAILABLE=true
    else
        print_warning "Python 3 is not installed"
        PYTHON_AVAILABLE=false
    fi
    
    # Check for NVIDIA Docker runtime
    if [ "$DOCKER_AVAILABLE" = true ]; then
        if docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi > /dev/null 2>&1; then
            print_status "NVIDIA Docker runtime detected"
            GPU_AVAILABLE=true
        else
            print_warning "NVIDIA Docker runtime not available"
            GPU_AVAILABLE=false
        fi
    else
        GPU_AVAILABLE=false
    fi
}

# Function to check required files
check_files() {
    echo "📁 Checking required files..."
    
    # Check for model file
    if [ -f "models/contador_yolo11n_270725_best.pt" ]; then
        print_status "Model file found"
    else
        print_warning "Model file 'models/contador_yolo11n_270725_best.pt' not found"
    fi
    
    # Check for input video
    if [ -f "media/operation_1920x1080.mp4" ]; then
        print_status "Input video 'operation_1920x1080.mp4' found"
    elif [ -f "media/normal_operation2.mp4" ]; then
        print_status "Input video 'normal_operation2.mp4' found"
    else
        print_warning "Input video not found in media/ directory"
    fi
    
    # Check for logo
    if [ -f "assets/LOGO_CE.png" ]; then
        print_status "Logo file found"
    else
        print_warning "Logo file 'assets/LOGO_CE.png' not found"
    fi
}

# Function to show menu
show_menu() {
    echo ""
    echo "🚀 Demo Varillas - Master Runner"
    echo "================================"
    echo ""
    echo "Available options:"
    echo ""
    
    if [ "$DOCKER_AVAILABLE" = true ] && [ "$COMPOSE_AVAILABLE" = true ]; then
        echo "1) 🐳 Run Original Implementation (Docker)"
        if [ "$IMAGE_EXISTS" = true ]; then
            echo "   📦 Image: Ready (no build needed)"
        else
            echo "   📦 Image: Not found (will build)"
        fi
        echo ""
        echo "2) ⚡ Run GStreamer Implementation (Docker + GPU)"
        if [ "$IMAGE_EXISTS" = true ]; then
            echo "   📦 Image: Ready (no build needed)"
        else
            echo "   📦 Image: Not found (will build)"
        fi
        echo ""
    fi
    
    if [ "$PYTHON_AVAILABLE" = true ]; then
        echo "3) 🐍 Run Local Development (Python)"
        echo ""
    fi
    
    echo "4) 🧹 Cleanup and Reset"
    echo "5) ℹ️  Show System Information"
    echo "6) ❌ Exit"
    echo ""
}

# Function to run original implementation
run_original() {
    echo "🐳 Running Original Implementation..."
    ./docker/run_original.sh
}

# Function to run GStreamer implementation
run_gstreamer() {
    echo "⚡ Running GStreamer Implementation..."
    ./docker/run_gstreamer.sh
}

# Function to run local development
run_local() {
    echo "🐍 Running Local Development..."
    ./docker/run_local.sh
}

# Function to cleanup
run_cleanup() {
    echo "🧹 Running Cleanup..."
    ./docker/cleanup.sh
}

# Function to show system info
show_system_info() {
    echo "ℹ️  System Information"
    echo "===================="
    echo ""
    echo "Docker: $DOCKER_AVAILABLE"
    echo "Docker Compose: $COMPOSE_AVAILABLE"
    echo "Python: $PYTHON_AVAILABLE"
    echo "GPU Support: $GPU_AVAILABLE"
    echo "Docker Image: $IMAGE_EXISTS"
    echo ""
    
    if [ "$PYTHON_AVAILABLE" = true ]; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        echo "Python Version: $PYTHON_VERSION"
    fi
    
    if [ "$DOCKER_AVAILABLE" = true ]; then
        DOCKER_VERSION=$(docker --version | cut -d' ' -f3 | cut -d',' -f1)
        echo "Docker Version: $DOCKER_VERSION"
    fi
    
    if [ "$COMPOSE_AVAILABLE" = true ]; then
        COMPOSE_VERSION=$(docker compose version --short 2>/dev/null || echo "Unknown")
        echo "Docker Compose Version: $COMPOSE_VERSION"
    fi
    
    echo ""
    echo "Available implementations:"
    if [ "$DOCKER_AVAILABLE" = true ] && [ "$COMPOSE_AVAILABLE" = true ]; then
        echo "  ✅ Docker-based implementations"
    else
        echo "  ❌ Docker-based implementations (Docker not available)"
    fi
    
    if [ "$PYTHON_AVAILABLE" = true ]; then
        echo "  ✅ Local development"
    else
        echo "  ❌ Local development (Python not available)"
    fi
    
    if [ "$GPU_AVAILABLE" = true ]; then
        echo "  ✅ GPU acceleration"
    else
        echo "  ❌ GPU acceleration (NVIDIA Docker not available)"
    fi
    
    if [ "$IMAGE_EXISTS" = true ]; then
        echo "  ✅ Docker image ready"
    else
        echo "  ⚠️  Docker image needs building"
    fi
}

# Main script
main() {
    # Change to project root directory
    cd "$PROJECT_ROOT"
    
    # Check prerequisites
    check_prerequisites
    
    # Check Docker images
    check_docker_images
    
    # Check required files
    check_files
    
    # Show menu and handle user input
    while true; do
        show_menu
        
        read -p "Enter your choice: " -n 1 -r
        echo ""
        
        case $REPLY in
            1)
                if [ "$DOCKER_AVAILABLE" = true ] && [ "$COMPOSE_AVAILABLE" = true ]; then
                    run_original
                else
                    print_error "Docker is not available"
                fi
                ;;
            2)
                if [ "$DOCKER_AVAILABLE" = true ] && [ "$COMPOSE_AVAILABLE" = true ]; then
                    run_gstreamer
                else
                    print_error "Docker is not available"
                fi
                ;;
            3)
                if [ "$PYTHON_AVAILABLE" = true ]; then
                    run_local
                else
                    print_error "Python is not available"
                fi
                ;;
            4)
                run_cleanup
                ;;
            5)
                show_system_info
                ;;
            6)
                print_info "Goodbye!"
                exit 0
                ;;
            *)
                print_error "Invalid choice. Please select 1-6."
                ;;
        esac
        
        echo ""
        read -p "Press Enter to continue..."
    done
}

# Run main function
main 