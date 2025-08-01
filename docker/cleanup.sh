#!/bin/bash

# Demo Varillas - Cleanup Script
# This script stops containers and cleans up resources

set -e  # Exit on any error

# Get the project root directory (two levels up from docker/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "🧹 Demo Varillas - Cleanup Script"
echo "================================="

# Change to project root directory
cd "$PROJECT_ROOT"

# Function to stop containers
stop_containers() {
    echo "🛑 Stopping Docker containers..."
    
    # Stop all demo-varillas containers
    docker compose down 2>/dev/null || echo "   No containers to stop."
    
    # Force stop any remaining containers with demo-varillas in the name
    docker ps -q --filter "name=demo-varillas" | xargs -r docker stop 2>/dev/null || echo "   No additional containers to stop."
    
    echo "✅ Containers stopped."
}

# Function to clean up Docker resources
cleanup_docker() {
    echo "🗑️  Cleaning up Docker resources..."
    
    # Remove stopped containers
    docker container prune -f 2>/dev/null || echo "   No containers to remove."
    
    # Remove unused images
    docker image prune -f 2>/dev/null || echo "   No images to remove."
    
    # Remove unused volumes
    docker volume prune -f 2>/dev/null || echo "   No volumes to remove."
    
    # Remove unused networks
    docker network prune -f 2>/dev/null || echo "   No networks to remove."
    
    echo "✅ Docker resources cleaned up."
}

# Function to remove Docker images
remove_images() {
    echo "🗑️  Removing Docker images..."
    
    # Ask user if they want to remove Docker images
    read -p "Do you want to remove Docker images? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Remove demo-varillas images
        docker images --format "table {{.Repository}}:{{.Tag}}" | grep "demo-varillas" | awk '{print $1}' | xargs -r docker rmi -f 2>/dev/null || echo "   No demo-varillas images to remove."
        echo "✅ Docker images removed."
    else
        echo "📦 Docker images preserved."
    fi
}

# Function to clean up output files
cleanup_output() {
    echo "📁 Cleaning up output files..."
    
    # Ask user if they want to clean output files
    read -p "Do you want to remove output files? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Remove output files but keep directories
        find output/ -type f -name "*.mp4" -delete 2>/dev/null || echo "   No output videos to remove."
        find logger/ -type f -name "*.png" -delete 2>/dev/null || echo "   No logger images to remove."
        find storage/ -type f -name "*.png" -delete 2>/dev/null || echo "   No storage images to remove."
        echo "✅ Output files removed."
    else
        echo "📁 Output files preserved."
    fi
}

# Function to reset X11 permissions
reset_x11() {
    echo "📺 Resetting X11 permissions..."
    xhost -local:docker 2>/dev/null || echo "   X11 permissions already reset."
    echo "✅ X11 permissions reset."
}

# Main cleanup process
echo "Starting cleanup process..."

# Stop containers
stop_containers

# Clean up Docker resources
cleanup_docker

# Remove Docker images (optional)
remove_images

# Clean up output files
cleanup_output

# Reset X11 permissions
reset_x11

echo ""
echo "✅ Cleanup completed successfully!"
echo ""
echo "📋 Summary:"
echo "   - Docker containers stopped"
echo "   - Docker resources cleaned up"
echo "   - Docker images handled"
echo "   - X11 permissions reset"
echo "   - Output files handled"
echo ""
echo "🚀 You can now run the application again with:"
echo "   ./docker/run_original.sh      # Original implementation"
echo "   ./docker/run_gstreamer.sh     # GStreamer implementation"
echo "   ./docker/run_local.sh         # Local development" 