#!/usr/bin/env python3
"""
Test script to verify MediaMTX integration
This script creates a simple test stream to MediaMTX
"""

import cv2
import numpy as np
import time
from datetime import datetime

def create_test_stream():
    """Create a simple test stream to MediaMTX"""
    print("🚀 Starting MediaMTX test stream...")
    
    # Video parameters
    fps = 15
    width = 800
    height = 600
    
    # Colors for cycling
    colors = [
        (0, 0, 255),    # Red
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
    ]
    
    # Create GStreamer pipeline for MediaMTX
    pipeline = (
        'appsrc ! videoconvert '
        '! video/x-raw,format=I420 '
        '! x264enc speed-preset=ultrafast bitrate=600 key-int-max=' + str(fps * 2) + ' '
        '! video/x-h264,profile=baseline '
        '! rtspclientsink location=rtsp://localhost:8888/test_stream'
    )
    
    try:
        # Create video writer
        out = cv2.VideoWriter(pipeline, cv2.CAP_GSTREAMER, 0, fps, (width, height), True)
        
        if not out.isOpened():
            print("❌ Failed to open MediaMTX stream writer")
            return False
        
        print("✅ MediaMTX stream writer created successfully")
        print("📺 Stream available at: rtsp://localhost:8888/test_stream")
        print("🌐 WebRTC available at: http://localhost:8888/webrtc_stream")
        
        # Generate test frames
        curcolor = 0
        start = time.time()
        
        while True:
            # Create a black frame
            frame = np.zeros((height, width, 3), np.uint8)
            
            # Create a colored rectangle
            color = colors[curcolor]
            curcolor = (curcolor + 1) % len(colors)
            
            # Fill top-left quadrant with color
            for y in range(0, int(frame.shape[0] / 2)):
                for x in range(0, int(frame.shape[1] / 2)):
                    frame[y][x] = color
            
            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Add stream info
            cv2.putText(frame, "MediaMTX Test Stream", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Write frame
            out.write(frame)
            print(f"📺 Frame written to MediaMTX: {timestamp}")
            
            # Show frame locally
            cv2.imshow('MediaMTX Test Stream', frame)
            
            # Frame rate control
            now = time.time()
            diff = (1 / fps) - (now - start)
            if diff > 0:
                time.sleep(diff)
            start = now
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n⏹️  Stream interrupted by user")
    except Exception as e:
        print(f"❌ Error during streaming: {e}")
    finally:
        out.release()
        cv2.destroyAllWindows()
        print("✅ Test stream completed")

if __name__ == "__main__":
    create_test_stream() 