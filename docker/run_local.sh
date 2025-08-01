#!/bin/bash

# Demo Varillas - Local Development
# This script runs the application locally without Docker

set -e  # Exit on any error

# Get the project root directory (two levels up from docker/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "🚀 Starting Demo Varillas - Local Development"
echo "============================================="

# Change to project root directory
cd "$PROJECT_ROOT"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "🐍 Python version: $PYTHON_VERSION"

# Check if virtual environment exists
if [ ! -d "env" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv env
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source env/bin/activate

# Install dependencies
echo "📥 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

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

# Ask user which implementation to run
echo ""
echo "Which implementation would you like to run?"
echo "1) Original OpenCV implementation (main.py)"
echo "2) GStreamer implementation (main2.py)"
echo ""
read -p "Enter your choice (1 or 2): " -n 1 -r
echo

if [[ $REPLY =~ ^[1]$ ]]; then
    echo "🐳 Running Original OpenCV Implementation..."
    python3 main.py
elif [[ $REPLY =~ ^[2]$ ]]; then
    echo "⚡ Running GStreamer Implementation..."
    python3 main2.py
else
    echo "❌ Invalid choice. Please run the script again and select 1 or 2."
    exit 1
fi

echo "✅ Demo Varillas - Local Development completed!"
echo "📁 Output files are saved in the output/ directory"
echo "📊 Logs are saved in the logger/ directory"

# Deactivate virtual environment
deactivate 