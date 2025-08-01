FROM ubuntu:20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies including GStreamer
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    # GStreamer packages with compatible versions
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    gstreamer1.0-x \
    gstreamer1.0-alsa \
    gstreamer1.0-gl \
    gstreamer1.0-gtk3 \
    gstreamer1.0-qt5 \
    gstreamer1.0-pulseaudio \
    # GStreamer development packages for Python bindings
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    # NVIDIA GStreamer plugins (for hardware acceleration)
    libnvidia-compute-470 \
    # OpenCV dependencies with compatible versions
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    gfortran \
    # Additional dependencies for GStreamer + OpenCV compatibility
    libgirepository1.0-dev \
    libcairo2-dev \
    pkg-config \
    # Additional system dependencies
    libglib2.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    # MediaMTX compilation dependencies
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Go 1.24.0 for MediaMTX compilation
RUN wget https://go.dev/dl/go1.24.0.linux-amd64.tar.gz && \
    tar -C /usr/local -xzf go1.24.0.linux-amd64.tar.gz && \
    rm go1.24.0.linux-amd64.tar.gz && \
    echo 'export PATH=$PATH:/usr/local/go/bin' >> /root/.bashrc

# Set Go environment variables
ENV PATH=$PATH:/usr/local/go/bin
ENV GOROOT=/usr/local/go
ENV GOPATH=/go

# Install MediaMTX
RUN cd /tmp && \
    git clone https://github.com/bluenviron/mediamtx && \
    cd mediamtx && \
    go generate ./... && \
    CGO_ENABLED=0 go build -o /usr/local/bin/mediamtx . && \
    cd / && \
    rm -rf /tmp/mediamtx

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with specific versions for compatibility
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create necessary directories if they don't exist
RUN mkdir -p models media output assets logger storage imgs storage/recordings

# Create MediaMTX configuration
RUN echo '# MediaMTX configuration for demo_varillas' > /app/mediamtx.yml && \
    echo 'paths:' >> /app/mediamtx.yml && \
    echo '  # Input stream from video file' >> /app/mediamtx.yml && \
    echo '  input_stream:' >> /app/mediamtx.yml && \
    echo '    source: /app/media/operation_1920x1080.mp4' >> /app/mediamtx.yml && \
    echo '' >> /app/mediamtx.yml && \
    echo '  # Detection output stream' >> /app/mediamtx.yml && \
    echo '  detection_output:' >> /app/mediamtx.yml && \
    echo '    source: rtsp://localhost:8554/detection' >> /app/mediamtx.yml && \
    echo '    record: true' >> /app/mediamtx.yml && \
    echo '    recordPath: /app/storage/recordings/' >> /app/mediamtx.yml && \
    echo '' >> /app/mediamtx.yml && \
    echo '  # WebRTC stream for browser viewing' >> /app/mediamtx.yml && \
    echo '  webrtc_stream:' >> /app/mediamtx.yml && \
    echo '    source: rtsp://localhost:8554/detection' >> /app/mediamtx.yml && \
    echo '    webrtc: true' >> /app/mediamtx.yml && \
    echo '' >> /app/mediamtx.yml && \
    echo '# API for control' >> /app/mediamtx.yml && \
    echo 'api: true' >> /app/mediamtx.yml && \
    echo 'apiAddress: :9997' >> /app/mediamtx.yml && \
    echo '' >> /app/mediamtx.yml && \
    echo '# Metrics' >> /app/mediamtx.yml && \
    echo 'metrics: true' >> /app/mediamtx.yml && \
    echo 'metricsAddress: :9998' >> /app/mediamtx.yml && \
    echo '' >> /app/mediamtx.yml && \
    echo '# RTSP settings' >> /app/mediamtx.yml && \
    echo 'rtspAddress: :8888' >> /app/mediamtx.yml && \
    echo 'rtspAddresses: [":8888", ":8889"]' >> /app/mediamtx.yml && \
    echo '' >> /app/mediamtx.yml && \
    echo '# RTMP settings' >> /app/mediamtx.yml && \
    echo 'rtmpAddress: :1935' >> /app/mediamtx.yml && \
    echo '' >> /app/mediamtx.yml && \
    echo '# HLS settings' >> /app/mediamtx.yml && \
    echo 'hlsAddress: :8888' >> /app/mediamtx.yml && \
    echo 'hlsAlwaysRemux: true' >> /app/mediamtx.yml

# Create startup script
RUN echo '#!/bin/bash' > /app/start_services.sh && \
    echo '' >> /app/start_services.sh && \
    echo '# Start MediaMTX in background' >> /app/start_services.sh && \
    echo 'echo "🚀 Starting MediaMTX server..."' >> /app/start_services.sh && \
    echo '/usr/local/bin/mediamtx /app/mediamtx.yml &' >> /app/start_services.sh && \
    echo 'MEDIAMTX_PID=$!' >> /app/start_services.sh && \
    echo '' >> /app/start_services.sh && \
    echo '# Wait for MediaMTX to start' >> /app/start_services.sh && \
    echo 'sleep 3' >> /app/start_services.sh && \
    echo '' >> /app/start_services.sh && \
    echo '# Check if MediaMTX is running' >> /app/start_services.sh && \
    echo 'if ! kill -0 $MEDIAMTX_PID 2>/dev/null; then' >> /app/start_services.sh && \
    echo '    echo "❌ Failed to start MediaMTX"' >> /app/start_services.sh && \
    echo '    exit 1' >> /app/start_services.sh && \
    echo 'fi' >> /app/start_services.sh && \
    echo '' >> /app/start_services.sh && \
    echo 'echo "✅ MediaMTX server started (PID: $MEDIAMTX_PID)"' >> /app/start_services.sh && \
    echo 'echo "📺 RTSP streams available at:"' >> /app/start_services.sh && \
    echo 'echo "   - rtsp://localhost:8888/input_stream (from operation_1920x1080.mp4)"' >> /app/start_services.sh && \
    echo 'echo "   - rtsp://localhost:8888/detection_output (detection results)"' >> /app/start_services.sh && \
    echo 'echo "   - rtsp://localhost:8888/webrtc_stream (browser view)"' >> /app/start_services.sh && \
    echo 'echo "🌐 WebRTC stream: http://localhost:8888/webrtc_stream"' >> /app/start_services.sh && \
    echo 'echo "📊 API: http://localhost:9997"' >> /app/start_services.sh && \
    echo 'echo "📈 Metrics: http://localhost:9998"' >> /app/start_services.sh && \
    echo '' >> /app/start_services.sh && \
    echo '# Start the main application' >> /app/start_services.sh && \
    echo 'echo "🚀 Starting demo_varillas application..."' >> /app/start_services.sh && \
    echo 'exec python3 main2.py' >> /app/start_services.sh

RUN chmod +x /app/start_services.sh

# Set the default command to start both services
CMD ["/app/start_services.sh"]
