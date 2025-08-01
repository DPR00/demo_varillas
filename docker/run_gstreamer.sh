#!/bin/bash

# Demo Varillas - GStreamer Implementation
# This script runs the GStreamer-enabled main2.py implementation using Docker

set -e  # Exit on any error

# Get the project root directory (two levels up from docker/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "🚀 Starting Demo Varillas - GStreamer Implementation (main2.py)"
echo "=============================================================="

# Change to project root directory
cd "$PROJECT_ROOT"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if docker compose is available
if ! docker compose version > /dev/null 2>&1; then
    echo "❌ Error: docker compose is not available. Please install Docker Compose and try again."
    exit 1
fi

# Check for NVIDIA Docker runtime
echo "🔍 Checking NVIDIA Docker runtime..."
if ! docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi > /dev/null 2>&1; then
    echo "⚠️  Warning: NVIDIA Docker runtime not available or GPU not detected."
    echo "   The application will run with CPU-only processing."
    echo "   For GPU acceleration, install NVIDIA Docker runtime:"
    echo "   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    echo ""
    read -p "Continue without GPU support? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Aborted by user."
        exit 1
    fi
else
    echo "✅ NVIDIA Docker runtime detected. GPU acceleration will be enabled."
fi

# Allow X11 connections for display
echo "📺 Setting up X11 display forwarding..."
xhost +local:docker 2>/dev/null || echo "⚠️  Warning: Could not set X11 permissions. GUI may not work."

# Check if required files exist
echo "🔍 Checking required files..."

# Check for model file
if [ ! -f "models/contador_yolo11n_270725_best.pt" ]; then
    echo "⚠️  Warning: Model file 'models/contador_yolo11n_270725_best.pt' not found."
    echo "   Please ensure your YOLO model is in the models/ directory."
fi

# Check for input video (check both possible names)
if [ ! -f "media/operation_1920x1080.mp4" ] && [ ! -f "media/normal_operation2.mp4" ]; then
    echo "⚠️  Warning: Input video not found in media/ directory."
    echo "   Expected: media/operation_1920x1080.mp4 or media/normal_operation2.mp4"
    echo "   Please ensure your input video is in the media/ directory."
fi

# Check for logo
if [ ! -f "assets/LOGO_CE.png" ]; then
    echo "⚠️  Warning: Logo file 'assets/LOGO_CE.png' not found."
    echo "   Please ensure your logo is in the assets/ directory."
fi

echo "✅ All checks completed."

# Check if Docker image already exists
echo "🔍 Checking if Docker image already exists..."
if docker images --format "table {{.Repository}}:{{.Tag}}" | grep -q "demo-varillas.*latest"; then
    echo "✅ Docker image found. Skipping build step."
    BUILD_FLAG=""
else
    echo "📦 Docker image not found. Building image..."
    BUILD_FLAG="--build"
fi

# Build and run the container
echo "🐳 Starting Docker container with GStreamer..."
echo "   This may take a few minutes on first run..."

docker compose up demo-varillas-gstreamer $BUILD_FLAG

echo "✅ Demo Varillas - GStreamer Implementation completed!"
echo "📁 Output files are saved in the output/ directory"
echo "📊 Logs are saved in the logger/ directory"
echo "⚡ GPU acceleration was enabled for better performance"
echo ""
echo "📺 MediaMTX Streams Available:"
echo "   - Detection: rtsp://localhost:8888/detection_output"
echo "   - WebRTC: http://localhost:8888/webrtc_stream"
echo "   - API: http://localhost:9997"
echo "   - Metrics: http://localhost:9998" 