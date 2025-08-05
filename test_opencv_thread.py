import cv2
import threading
import queue
import time
import numpy as np
from ultralytics import YOLO  # pip install ultralytics
from torch import cuda as t_cuda
from torch import device as t_device
from scripts import CameraParameters, get_data, Tracker, plot_historic, read_yaml_file, get_positions, Logger
import os
# ===== Configuración Global =====
# RTSP_URL = "rtsp://admin:IA+T3cCAM@192.168.18.3/Streaming/Channels/101?transportmode=unicast"

dir_path = os.path.dirname(os.path.abspath(__file__))
data = get_data(dir_path)
RTSP_URL = data['video_path'] #"media/operation_1920x1080.mp4"  # O tu URL RTSP
MODEL_PATH = data['model_path'] #"models/contador_yolo11n_030825.pt"
BUFFER_SIZE = 1  # Tamaño del buffer para baja latencia
WIDTH, HEIGHT = 1920, 1080

# Colas para comunicación entre hilos
raw_frame_queue = queue.Queue(maxsize=2)       # Frames sin procesar
processed_frame_queue = queue.Queue(maxsize=2)  # Frames procesados con detecciones
stop_event = threading.Event()                 # Señal de parada para todos los hilos

# ===== Hilo 1: Captura de Video =====
def video_capture_thread():
    cap = cv2.VideoCapture(RTSP_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, BUFFER_SIZE)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H','2','6','4'))  # Códec H.264

    if not cap.isOpened():
        print("Error: No se pudo abrir la fuente de video")
        stop_event.set()
        return

    print("Hilo de captura iniciado")
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Error de lectura de frame")
            time.sleep(0.1)
            # stop_event.set()
            continue
        
        # Limpiar cola si está llena para mantener solo el frame más reciente
        if raw_frame_queue.full():
            try:
                raw_frame_queue.get_nowait()
            except queue.Empty:
                pass
        ## DELETE THIS FOR THE REAL DEMO (THIS LINE IS ONLY TO SIMULATE 30 FPS WHEN READING A SAVED VIDEO)
        time.sleep(0.033)

        raw_frame_queue.put(frame)
    cap.release()
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
            frame = raw_frame_queue.get(timeout=0.5)

            # ROI frame
            roi_frame = frame[cam_params.y : cam_params.y + cam_params.h,
                            cam_params.x : cam_params.x + cam_params.w]            # Realizar detecciones con YOLO
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
            
        except queue.Empty:
            pass  # No hay frames disponibles, continuar
        except Exception as e:
            print(f"Error en procesamiento: {str(e)}")
    
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
            display_frame = processed_frame #cv2.resize(processed_frame, (1280, 720))
            
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
    
    print("Sistema terminado")

if __name__ == "__main__":
    main()