#!/bin/bash

# Start MediaMTX in background
echo "🚀 Starting MediaMTX server..."
/usr/local/bin/mediamtx /app/mediamtx.yml &
MEDIAMTX_PID=$!

# Wait for MediaMTX to start
sleep 3

# Check if MediaMTX is running
if ! kill -0 $MEDIAMTX_PID 2>/dev/null; then
    echo "❌ Failed to start MediaMTX"
    exit 1
fi

echo "✅ MediaMTX server started (PID: $MEDIAMTX_PID)"
echo "📺 RTSP streams available at:"
echo "   - rtsp://localhost:8888/input_stream (from operation_1920x1080.mp4)"
echo "   - rtsp://localhost:8888/detection_output (detection results)"
echo "   - rtsp://localhost:8888/webrtc_stream (browser view)"
echo "🌐 WebRTC stream: http://localhost:8888/webrtc_stream"
echo "📊 API: http://localhost:9997"
echo "📈 Metrics: http://localhost:9998"

# Start the main application
echo "🚀 Starting demo_varillas application..."
exec python3 main2.py 