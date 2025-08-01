#!/bin/bash

# MediaMTX Test Script
# This script tests the MediaMTX integration

set -e  # Exit on any error

# Get the project root directory (two levels up from docker/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "🧪 Testing MediaMTX Integration"
echo "==============================="

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

# Check if required files exist
echo "🔍 Checking required files..."

# Check for test script
if [ ! -f "test_mediamtx.py" ]; then
    echo "❌ Error: test_mediamtx.py not found."
    exit 1
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

# Start the container with MediaMTX
echo "🐳 Starting Docker container with MediaMTX..."
echo "   This may take a few minutes on first run..."

# Run the container in background
docker compose up demo-varillas-gstreamer $BUILD_FLAG -d

# Wait for MediaMTX to start
echo "⏳ Waiting for MediaMTX to start..."
sleep 10

# Test MediaMTX API
echo "🔍 Testing MediaMTX API..."
if curl -s http://localhost:9997/v3/paths/list > /dev/null; then
    echo "✅ MediaMTX API is responding"
else
    echo "❌ MediaMTX API is not responding"
fi

# Test RTSP endpoint
echo "🔍 Testing RTSP endpoint..."
if curl -s http://localhost:8888 > /dev/null; then
    echo "✅ RTSP endpoint is responding"
else
    echo "❌ RTSP endpoint is not responding"
fi

# Show available streams
echo ""
echo "📺 Available MediaMTX Streams:"
echo "   - RTSP: rtsp://localhost:8888/detection_output"
echo "   - WebRTC: http://localhost:8888/webrtc_stream"
echo "   - API: http://localhost:9997"
echo "   - Metrics: http://localhost:9998"

# Instructions for testing
echo ""
echo "🧪 To test the streams:"
echo "   1. Open VLC and play: rtsp://localhost:8888/detection_output"
echo "   2. Open browser and go to: http://localhost:8888/webrtc_stream"
echo "   3. Check API status: curl http://localhost:9997/v3/paths/list"

# Stop the container
echo ""
echo "🛑 Stopping container..."
docker compose down demo-varillas-gstreamer

echo "✅ MediaMTX test completed!" 