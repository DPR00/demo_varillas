import av
import cv2
import threading
import queue
import time
import numpy as np
from ultralytics import YOLO
from torch import cuda as t_cuda
from torch import device as t_device
from av.error import FFmpegError
# Importaciones de tus scripts personalizados
from scripts import CameraParameters, get_data, Tracker, plot_historic, get_positions, Logger
import os

# ===== Configuración Global =====
dir_path = os.path.dirname(os.path.abspath(__file__))
data = get_data(dir_path)
RTSP_URL = data['video_path']
MODEL_PATH = data['model_path']
BUFFER_SIZE = 1
WIDTH, HEIGHT = 1920, 1080

# Opciones optimizadas para conexiones inestables (PyAV)
FFMPEG_OPTIONS = {
    'rtsp_transport': 'tcp',
    'fflags': 'nobuffer',
    'flags': 'low_delay',
    'max_delay': '500000',
    'analyzeduration': '100000',
    'probesize': '1024',
    'tune': 'zerolatency',
    'framedrop': '1',
    'avioflags': 'direct',
    'flush_packets': '1',
    'timeout': '5000000',
    'reconnect': '1',
    'reconnect_at_eof': '1',
    'reconnect_streamed': '1',
    'reconnect_delay_max': '5',
    'stimeout': '5000000',
    'heartbeat_interval': '10'
}

# Colas para comunicación entre hilos
raw_frame_queue = queue.Queue(maxsize=2)       # Frames sin procesar
processed_frame_queue = queue.Queue(maxsize=2)  # Frames procesados con detecciones
stop_event = threading.Event()                 # Señal de parada para todos los hilos

# ===== Hilo 1: Captura de Video con PyAV =====
def video_capture_thread():
    print("Hilo de captura iniciado (PyAV)")
    last_frame = None
    
    while not stop_event.is_set():
        try:
            # Abrir el stream con PyAV
            container = av.open(RTSP_URL, options=FFMPEG_OPTIONS)
            stream = container.streams.video[0]
            print(f"Conexión RTSP establecida: {RTSP_URL}")
            
            for packet in container.demux(stream):
                if stop_event.is_set():
                    break
                    
                for frame in packet.decode():
                    if stop_event.is_set():
                        break
                        
                    # Convertir frame a array de numpy (BGR para OpenCV)
                    img = frame.to_ndarray(format='bgr24')
                    last_frame = img
                    
                    # Limpiar cola si está llena para mantener solo el frame más reciente
                    if raw_frame_queue.full():
                        try:
                            raw_frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    
                    raw_frame_queue.put(img)
        
        except FFmpegError as e:
            print(f"Error de conexión (PyAV): {e}")
            # Mostrar último frame durante la reconexión
            if last_frame is not None:
                # Limpiar cola si está llena
                if raw_frame_queue.full():
                    try:
                        raw_frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                raw_frame_queue.put(last_frame.copy())  # Enviamos una copia del último frame
            print("Reintentando conexión en 2 segundos...")
            time.sleep(2)
            
        except Exception as e:
            print(f"Error inesperado en captura (PyAV): {e}")
            # Mostrar último frame durante la reconexión
            if last_frame is not None:
                if raw_frame_queue.full():
                    try:
                        raw_frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                raw_frame_queue.put(last_frame.copy())
            print("Reintentando conexión en 5 segundos...")
            time.sleep(5)
    
    print("Hilo de captura terminado")

# ===== Hilo 2: Procesamiento con YOLO =====
def processing_thread():
    # variables
    frame_count = 0
    track_id = 1
    tracking_objects = {}
    center_points_prev_frame = []
    list_counter = [] # For plot historic
    actuator_initial_pos = (0,0)
    min_track = 0
    prev_size = -1
    max_key = -1
    counted = False
    stored_list = False
    actuator_moving = False
    rod_count = 0
    counted_track_ids = set()
    video_writer = None
    if data['generate_video']:
        output_path = data['output_path']
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30.0  # Asumir 30 FPS 
        frame_size = (data['roi_width'], data['roi_height'])
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        
        if not video_writer.isOpened():
            print(f"[ERROR] Could not initialize video writer for {output_path}")
            video_writer = None
        else:
            print(f"Video writer initialized: {output_path} at {fps} FPS, size {frame_size}")
    
    cam_params = CameraParameters(WIDTH, HEIGHT,
                                  x = data['x_init'], y = data['y_init'],
                                  w = data['roi_width'], h = data['roi_height'])
    cam_params.update_limits(data['counter_init'], data['counter_end'], data['counter_line'])

    model = YOLO(MODEL_PATH)
    print(f"Modelo YOLO cargado: {MODEL_PATH}")
    device = t_device("cuda" if t_cuda.is_available() else "cpu")
    model.to(device)
    print(f"Usando dispositivo: {device}")

    # Logger
    storage_path = data['storage_path'] if data['storage_data'] else None
    logger = Logger(output_dir = data['logger_path'], storage_path = storage_path)
    roi_frame = None
    print("Hilo de procesamiento iniciado")
    while not stop_event.is_set():
        try:
            # Obtener el último frame disponible (esperar máximo 0.5s)
            full_frame = raw_frame_queue.get(timeout=0.5)

            # ROI frame
            roi_frame = full_frame[cam_params.y : cam_params.y + cam_params.h,
                                  cam_params.x : cam_params.x + cam_params.w]
            
            # Realizar detecciones con YOLO
            detections = model(roi_frame, verbose=False, stream=False)

            # Procesar resultados
            if prev_size == len(list_counter):
                plot_historic(roi_frame, list_counter, data['logo'])

            center_points_cur_frame, actuator_pos = get_positions(detections,
                                                                data['min_confidence'],
                                                                data['actuator_data'])
            
            # Limpiar cola si está llena para mantener solo el frame más reciente
            if processed_frame_queue.full():
                try:
                    processed_frame_queue.get_nowait()
                except queue.Empty:
                    pass
            
            sorted_center_points_cur_frame = sorted(center_points_cur_frame, key = lambda point: point.pos_x)
            sorted_center_points_cur_frame.reverse()

            tracker = Tracker(sorted_center_points_cur_frame, roi_frame, cam_params, debug=data['debug'])
            tracker.update_params(track_id, tracking_objects, center_points_prev_frame, rod_count, counted_track_ids)
            track_id, tracking_objects, center_points_prev_frame, rod_count, counted_track_ids = tracker.track()
            tracker.plot_count()

            if data['debug']:
                logger.log(roi_frame, frame_count)
            frame_count += 1
            prev_size = len(list_counter)

            processed_frame_queue.put(roi_frame)

            if video_writer is not None:
                # Clonar el frame para evitar modificaciones posteriores
                frame_to_write = roi_frame.copy()
                video_writer.write(frame_to_write)
            
        except queue.Empty:
            pass  # No hay frames disponibles, continuar
        except Exception as e:
            print(f"Error en procesamiento: {str(e)}")
            import traceback
            traceback.print_exc()

    if video_writer is not None:
        video_writer.release()
        print("Video writer released")
    print("Hilo de procesamiento terminado")

# ===== Hilo 3: Visualización =====
def display_thread():
    print("Hilo de visualización iniciado")
    cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)
    
    last_frame = None
    fps = 0
    frame_count = 0
    last_time = time.time()
    
    while not stop_event.is_set():
        start_time = time.perf_counter()
        
        try:
            # Obtener el último frame procesado
            processed_frame = processed_frame_queue.get(timeout=0.5)
            last_frame = processed_frame
            
            # Calcular FPS
            frame_count += 1
            current_time = time.time()
            elapsed = current_time - last_time
            
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                last_time = current_time
        except queue.Empty:
            processed_frame = last_frame  # Usar último frame disponible
            if processed_frame is None:
                continue
        
        # Mostrar información de rendimiento
        if processed_frame is not None:
            display_frame = processed_frame
            
            # Mostrar FPS y estado
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (400, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Mostrar tamaño de colas
            cv2.putText(display_frame, f"Captura: {raw_frame_queue.qsize()}/2", (400, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Procesados: {processed_frame_queue.qsize()}/2", (400, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Mostrar frame
            cv2.imshow("YOLO Detection", display_frame)
        
        # Manejar entrada de teclado
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            stop_event.set()
            break
    
    cv2.destroyAllWindows()
    print("Hilo de visualización terminado")

# ===== Función Principal =====
def main():
    # Crear e iniciar hilos
    threads = [
        threading.Thread(target=video_capture_thread, daemon=True),
        threading.Thread(target=processing_thread, daemon=True),
        threading.Thread(target=display_thread, daemon=True)
    ]
    
    for t in threads:
        t.start()
    
    # Esperar a que todos los hilos terminen
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(0.5)
    except KeyboardInterrupt:
        stop_event.set()
        print("Deteniendo todos los hilos...")
    
    print("Sistema terminado")

if __name__ == "__main__":
    main()