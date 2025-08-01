import cv2
import gi
import numpy as np
import threading
import time
from typing import Optional, Tuple, Callable

# Initialize GStreamer with proper error handling
try:
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst, GLib
    GSTREAMER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: GStreamer not available: {e}")
    GSTREAMER_AVAILABLE = False

class GStreamerCapture:
    """
    GStreamer-based video capture class that can handle various input sources
    with hardware acceleration and real-time processing.
    """
    
    def __init__(self, source: str, width: int = 1920, height: int = 1080, 
                 fps: int = 20, use_hw_accel: bool = True):
        """
        Initialize GStreamer capture.
        
        Args:
            source: Input source (file path, RTSP URL, device path, etc.)
            width: Frame width
            height: Frame height
            fps: Target FPS
            use_hw_accel: Enable hardware acceleration
        """
        if not GSTREAMER_AVAILABLE:
            raise ImportError("GStreamer is not available. Please install GStreamer and PyGObject.")
        
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.use_hw_accel = use_hw_accel
        
        # Initialize GStreamer
        Gst.init(None)
        
        # Pipeline components
        self.pipeline = None
        self.appsink = None
        self.loop = None
        
        # Threading
        self.running = False
        self.frame_available = threading.Event()
        self.current_frame = None
        
    def _build_pipeline(self) -> str:
        """Build GStreamer pipeline based on source type."""
        
        # Hardware acceleration elements (only if available and requested)
        if self.use_hw_accel:
            try:
                # Test if NVIDIA plugins are available
                test_pipeline = Gst.parse_launch("nvv4l2decoder ! fakesink")
                test_pipeline.set_state(Gst.State.NULL)
                hw_decoder = "nvv4l2decoder"
                hw_converter = "nvvideoconvert"
            except:
                print("Warning: NVIDIA GStreamer plugins not available, using CPU fallback")
                hw_decoder = "decodebin"
                hw_converter = "videoconvert"
        else:
            hw_decoder = "decodebin"
            hw_converter = "videoconvert"
        
        # Determine source type and build appropriate pipeline
        if self.source.startswith(('rtsp://', 'rtp://', 'udp://')):
            # Network stream
            pipeline_str = f"""
                rtspsrc location={self.source} latency=0 !
                rtph264depay !
                h264parse !
                {hw_decoder} !
                {hw_converter} !
                video/x-raw,format=BGR,width={self.width},height={self.height},framerate={self.fps}/1 !
                appsink name=sink sync=false max-buffers=1 drop=true
            """
        elif self.source.startswith('/dev/'):
            # USB camera or device
            pipeline_str = f"""
                v4l2src device={self.source} !
                video/x-raw,format=YUY2,width={self.width},height={self.height},framerate={self.fps}/1 !
                videoconvert !
                video/x-raw,format=BGR !
                appsink name=sink sync=false max-buffers=1 drop=true
            """
        else:
            # File input
            pipeline_str = f"""
                filesrc location={self.source} !
                decodebin !
                {hw_converter} !
                video/x-raw,format=BGR,width={self.width},height={self.height},framerate={self.fps}/1 !
                appsink name=sink sync=false max-buffers=1 drop=true
            """
        
        return pipeline_str.strip()
    
    def _on_new_sample(self, sink) -> Gst.FlowReturn:
        """Callback for new frame from GStreamer pipeline."""
        try:
            sample = sink.emit('pull-sample')
            buffer = sample.get_buffer()
            caps = sample.get_caps()
            
            # Get frame data
            success, map_info = buffer.map(Gst.MapFlags.READ)
            if not success:
                return Gst.FlowReturn.ERROR
            
            # Convert to numpy array
            frame_data = map_info.data
            buffer.unmap(map_info)
            
            # Reshape to image dimensions
            frame = np.ndarray(
                shape=(self.height, self.width, 3),
                dtype=np.uint8,
                buffer=frame_data
            ).copy()  # Make a copy to avoid memory issues
            
            self.current_frame = frame
            self.frame_available.set()
            
            return Gst.FlowReturn.OK
        except Exception as e:
            print(f"Error in GStreamer callback: {e}")
            return Gst.FlowReturn.ERROR
    
    def start(self) -> bool:
        """Start the GStreamer pipeline."""
        try:
            # Build pipeline
            pipeline_str = self._build_pipeline()
            print(f"Starting GStreamer pipeline: {pipeline_str}")
            self.pipeline = Gst.parse_launch(pipeline_str)
            
            # Get appsink
            self.appsink = self.pipeline.get_by_name('sink')
            if not self.appsink:
                raise RuntimeError("Could not find appsink in pipeline")
            
            self.appsink.connect('new-sample', self._on_new_sample, None)
            
            # Start pipeline
            ret = self.pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("Failed to start GStreamer pipeline")
            
            # Start main loop in separate thread
            self.running = True
            self.loop = GLib.MainLoop()
            thread = threading.Thread(target=self.loop.run, daemon=True)
            thread.start()
            
            # Wait a bit for pipeline to start
            time.sleep(0.5)
            
            return True
            
        except Exception as e:
            print(f"Error starting GStreamer pipeline: {e}")
            return False
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the pipeline.
        
        Returns:
            Tuple of (success, frame)
        """
        if not self.running:
            return False, None
        
        # Wait for frame with timeout
        if self.frame_available.wait(timeout=1.0):
            self.frame_available.clear()
            return True, self.current_frame
        
        return False, None
    
    def release(self):
        """Release resources."""
        self.running = False
        
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        
        if self.loop:
            self.loop.quit()
    
    def get(self, prop_id: int) -> float:
        """Get pipeline property (compatible with cv2.VideoCapture)."""
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        elif prop_id == cv2.CAP_PROP_FPS:
            return float(self.fps)
        else:
            return 0.0

class GStreamerWriter:
    """
    GStreamer-based video writer with hardware acceleration.
    """
    
    def __init__(self, filename: str, width: int, height: int, fps: int = 30,
                 use_hw_accel: bool = True):
        """
        Initialize GStreamer writer.
        
        Args:
            filename: Output file path
            width: Frame width
            height: Frame height
            fps: Target FPS
            use_hw_accel: Enable hardware acceleration
        """
        if not GSTREAMER_AVAILABLE:
            raise ImportError("GStreamer is not available. Please install GStreamer and PyGObject.")
        
        self.filename = filename
        self.width = width
        self.height = height
        self.fps = fps
        self.use_hw_accel = use_hw_accel
        
        # Initialize GStreamer
        Gst.init(None)
        
        # Pipeline components
        self.pipeline = None
        self.appsrc = None
        
        # Hardware acceleration elements (only if available and requested)
        if self.use_hw_accel:
            try:
                # Test if NVIDIA plugins are available
                test_pipeline = Gst.parse_launch("nvv4l2h264enc ! fakesink")
                test_pipeline.set_state(Gst.State.NULL)
                hw_encoder = "nvv4l2h264enc"
                hw_converter = "nvvideoconvert"
            except:
                print("Warning: NVIDIA GStreamer plugins not available, using CPU fallback")
                hw_encoder = "x264enc"
                hw_converter = "videoconvert"
        else:
            hw_encoder = "x264enc"
            hw_converter = "videoconvert"
        
        # Build pipeline
        pipeline_str = f"""
            appsrc name=src format=time is-live=true do-timestamp=true !
            video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1 !
            {hw_converter} !
            video/x-raw,format=I420 !
            {hw_encoder} !
            h264parse !
            mp4mux !
            filesink location={filename}
        """
        
        self.pipeline = Gst.parse_launch(pipeline_str.strip())
        self.appsrc = self.pipeline.get_by_name('src')
        
        # Start pipeline
        self.pipeline.set_state(Gst.State.PLAYING)
    
    def write(self, frame: np.ndarray):
        """Write a frame to the output file."""
        if self.appsrc is None:
            return False
        
        try:
            # Convert frame to bytes
            frame_bytes = frame.tobytes()
            
            # Create buffer
            buffer = Gst.Buffer.new_wrapped(frame_bytes)
            
            # Push to pipeline
            ret = self.appsrc.emit('push-buffer', buffer)
            return ret == Gst.FlowReturn.OK
        except Exception as e:
            print(f"Error writing frame: {e}")
            return False
    
    def release(self):
        """Release resources."""
        if self.appsrc:
            self.appsrc.emit('end-of-stream')
        
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)

# Factory functions for easy integration
def create_gstreamer_capture(source: str, **kwargs) -> GStreamerCapture:
    """Create a GStreamer capture instance."""
    return GStreamerCapture(source, **kwargs)

def create_gstreamer_writer(filename: str, width: int, height: int, **kwargs) -> GStreamerWriter:
    """Create a GStreamer writer instance."""
    return GStreamerWriter(filename, width, height, **kwargs)

# Fallback function for when GStreamer is not available
def create_opencv_capture(source: str, **kwargs):
    """Fallback to OpenCV capture when GStreamer is not available."""
    print("Warning: Using OpenCV capture as GStreamer is not available")
    return cv2.VideoCapture(source)

def create_opencv_writer(filename: str, width: int, height: int, fps: int = 30, **kwargs):
    """Fallback to OpenCV writer when GStreamer is not available."""
    print("Warning: Using OpenCV writer as GStreamer is not available")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(filename, fourcc, fps, (width, height)) 