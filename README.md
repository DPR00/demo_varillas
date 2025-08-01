# Demo Varillas - Computer Vision Application

This project implements a computer vision system for counting rods/varillas using YOLO object detection and OpenCV. It now includes GStreamer support for enhanced video processing capabilities.

## Features

- YOLO-based object detection for rod counting
- Real-time video processing
- Tracking and counting algorithms
- Configurable ROI (Region of Interest)
- Video output generation
- Logging and storage capabilities
- **NEW**: GStreamer integration with hardware acceleration
- **NEW**: Support for RTSP streams, USB cameras, and IP cameras

## Prerequisites

- Docker and Docker Compose installed
- NVIDIA Docker runtime (optional, for GPU acceleration)

## Project Structure

```
demo_varillas/
├── assets/          # Images and logos
├── config/          # Configuration files
├── dataset/         # Training datasets
├── docker/          # Docker configuration and scripts
│   ├── Dockerfile
│   ├── run_original.sh      # Original implementation runner
│   ├── run_gstreamer.sh     # GStreamer implementation runner
│   ├── run_local.sh         # Local development runner
│   ├── cleanup.sh           # Cleanup script
│   └── run_all.sh           # Master runner with menu
├── media/           # Input videos
├── models/          # YOLO model files
├── output/          # Generated videos
├── scripts/         # Python modules
│   └── gstreamer_utils.py  # GStreamer utilities
├── storage/         # Storage for processed data
├── tests/           # Test files
├── main.py          # Original OpenCV-based implementation
└── main2.py         # GStreamer-enabled implementation
```

## Quick Start

### Option 1: Master Runner (Recommended)

```bash
# Run the master script with interactive menu
./docker/run_all.sh
```

### Option 2: Individual Scripts

#### Original Implementation (main.py)
```bash
# Run the original application
./docker/run_original.sh

# Or run directly
docker compose up demo-varillas --build
```

#### GStreamer Implementation (main2.py)
```bash
# Run the GStreamer-enabled application
./docker/run_gstreamer.sh

# Or run directly with GPU support
docker compose up demo-varillas-gstreamer --build
```

#### Local Development
```bash
# Run locally without Docker
./docker/run_local.sh
```

### Option 3: Manual Docker Commands

```bash
# Build and start the original application
docker compose up demo-varillas --build

# Build and start the GStreamer application
docker compose up demo-varillas-gstreamer --build

# Or run directly with GPU support
docker run -it --rm \
  --gpus all \
  -v $(pwd)/media:/app/media:ro \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/assets:/app/assets:ro \
  -v $(pwd)/config:/app/config:ro \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/logger:/app/logger \
  -v $(pwd)/storage:/app/storage \
  -v $(pwd)/scripts:/app/scripts:ro \
  --network host \
  demo-varillas-gstreamer
```

## Configuration

The application is configured through `config/params.yaml`:

### Basic Configuration
- **Model**: Specify the YOLO model file in the `models/` directory
- **Input Video**: Set the input video file in the `media/` directory
- **ROI Settings**: Configure the region of interest for detection
- **Tracking Parameters**: Adjust confidence thresholds and tracking settings
- **Output Settings**: Control video generation and logging

### GStreamer Configuration (for main2.py)
```yaml
gstreamer:
  use_hw_accel: true          # Enable hardware acceleration
  input_source: "media/operation_1920x1080.mp4"  # File path, RTSP URL, or device
  width: 1920                 # Frame width
  height: 1080                # Frame height
  fps: 30                     # Target FPS
  latency: 0                  # For real-time streams
```

## Input Sources

The GStreamer implementation supports various input sources:

### File Input
```yaml
gstreamer:
  input_source: "media/operation_1920x1080.mp4"
```

### RTSP Stream
```yaml
gstreamer:
  input_source: "rtsp://192.168.1.100:554/stream"
```

### USB Camera
```yaml
gstreamer:
  input_source: "/dev/video0"
```

### IP Camera
```yaml
gstreamer:
  input_source: "http://192.168.1.101:8080/video"
```

## Required Files

Before running the application, ensure you have:

1. **YOLO Model**: Place your trained YOLO model (`.pt` file) in the `models/` directory
2. **Input Video**: Place your input video file in the `media/` directory
3. **Logo**: Place your logo image in the `assets/` directory

## Development

### Local Development (without Docker)

```bash
# Install Python dependencies
pip install -r requirements.txt

# Run the original implementation
python main.py

# Run the GStreamer implementation
python main2.py
```

### Building for Production

```bash
# Build optimized production image
docker build -f docker/Dockerfile -t demo-varillas:prod .

# Run production container with GStreamer
docker run -d --name demo-varillas-prod \
  --gpus all \
  -v $(pwd)/media:/app/media:ro \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/assets:/app/assets:ro \
  -v $(pwd)/config:/app/config:ro \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/logger:/app/logger \
  -v $(pwd)/storage:/app/storage \
  -v $(pwd)/scripts:/app/scripts:ro \
  demo-varillas:prod python3 main2.py
```

## Performance Comparison

| Feature | main.py (OpenCV) | main2.py (GStreamer) |
|---------|------------------|----------------------|
| Hardware Acceleration | Limited | Full NVIDIA GPU support |
| Input Sources | Files only | Files, RTSP, USB, IP cameras |
| Real-time Processing | Moderate | High performance |
| Latency | Higher | Lower |
| Memory Usage | Standard | Optimized |

## Troubleshooting

### Display Issues
If you encounter display issues when running with GUI:

```bash
# Allow X11 forwarding
xhost +local:docker

# Run with X11 socket mounted
docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  [other volume mounts...] \
  demo-varillas
```

### GPU Issues
If GPU acceleration isn't working:

1. Ensure NVIDIA Docker runtime is installed
2. Check that your GPU drivers are up to date
3. Verify CUDA compatibility with your PyTorch version
4. For GStreamer, ensure NVIDIA plugins are installed

### GStreamer Issues
If GStreamer pipeline fails:

1. Check input source path/URL
2. Verify GStreamer plugins are installed
3. Test pipeline manually: `gst-launch-1.0 [pipeline]`
4. Check hardware acceleration compatibility

### Permission Issues
If you encounter permission issues with mounted volumes:

```bash
# Fix permissions
sudo chown -R $USER:$USER ./output ./logger ./storage
```

### Script Issues
If the shell scripts don't work:

```bash
# Make scripts executable
chmod +x docker/*.sh

# Run from project root directory
cd /path/to/demo_varillas
./docker/run_all.sh
```

### Docker Compose Issues
If you encounter Docker Compose issues:

```bash
# Check Docker Compose version
docker compose version

# If using older Docker Compose, you may need to install it separately
# For newer Docker versions, Docker Compose is included as 'docker compose'
```
