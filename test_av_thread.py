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
import serial
from scripts import CameraParameters, get_data, Tracker, plot_historic, get_positions, Logger, handle_actuator
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

SERIAL_PORT = 'COM3'  # Linux: '/dev/ttyUSB0', Windows: 'COM3'
BAUD_RATE = 115200
TIMEOUT = 1  # segundos

data['flow_data'] = dict(serial_counter=0, direction=1, direction_raw="", frame=None, direction_time=0, frame_time=0)

def read_serial_thread(data_params: dict):
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)
    except serial.SerialException as e:
        print(f"Error al abrir {SERIAL_PORT}: {e}")
        return

    # Dar tiempo para que el ESP32 reinicie y desechar logs de arranque
    time.sleep(2)
    ser.reset_input_buffer()

    print(f"Leyendo bits desde {SERIAL_PORT} a {BAUD_RATE} baudios...")
    serial_data = dict()
    direction = 1
    try:
        while not stop_event.is_set():
            raw = ser.readline()
            if not raw:
                continue
            try:
                line = raw.decode('utf-8').strip()
            except UnicodeDecodeError:
                continue

            # Si la línea consiste en exactamente 4 caracteres de '0' o '1', la mostramos
            if len(line) == 2 and all(c in '01' for c in line):
                if line == '01':
                    direction = -1
                elif line == '11':
                    direction = 0
                else:
                    direction = 1 # Default direction
            # En otro caso, simplemente ignoramos la línea
            serial_data['direction'] = direction
            serial_data['direction_raw'] = line
            serial_data['direction_time'] = time.time()
            data_params['flow_data'].update(**serial_data)
            data_params['flow_data']['serial_counter'] = data_params['flow_data'].get("serial_counter", 0) + 1
            # print(f"raw: {line}, direction: {direction}, time={serial_data['direction_time']}")

    except KeyboardInterrupt:
        print("Deteniendo lectura...")
    finally:
        ser.close()

# ===== Hilo 1: Captura de Video con PyAV y Serial =====
def video_capture_thread(data_params: dict):
    print("Hilo de captura iniciado (PyAV)")
    last_frame = None
    last_direction = 1  # Dirección por defecto (izquierda a derecha)
    raw_data_motor = b''
    
    while not stop_event.is_set():
        data_to_send = {}
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
                    
                    # Leer datos del puerto serial
                    
                    
                    # Limpiar cola si está llena para mantener solo el frame más reciente
                    # if raw_frame_queue.full():
                    #     try:
                    #         raw_frame_queue.get_nowait()
                    #     except queue.Empty:
                    #         pass
                    
                    # # Enviar frame y dirección
                    # raw_frame_queue.put({
                    #     'frame': img,
                    #     'direction': last_direction
                    # })
                    data_params['flow_data']["frame"] = last_frame.copy()
                    data_params['flow_data']["frame_time"] = time.time()
                    data_params['flow_data']['serial_counter'] = 0
        
        except FFmpegError as e:
            print(f"Error de conexión (PyAV): {e}")
            
            # Mostrar último frame durante la reconexión
            if last_frame is not None:
                # if raw_frame_queue.full():
                #     try:
                #         raw_frame_queue.get_nowait()
                #     except queue.Empty:
                #         pass
                # raw_frame_queue.put({
                #     'frame': last_frame.copy(),
                #     'direction': last_direction
                # })
                data_params['flow_data']["frame"] = last_frame.copy()
                data_params['flow_data']["frame_time"] = time.time()
            print("Reintentando conexión en 2 segundos...")
            time.sleep(2)
            
        except Exception as e:
            print(f"Error inesperado en captura (PyAV): {e}")
            # Leer datos del puerto serial durante la reconexión
            
            
            # Mostrar último frame durante la reconexión
            if last_frame is not None:
                # if raw_frame_queue.full():
                #     try:
                #         raw_frame_queue.get_nowait()
                #     except queue.Empty:
                #         pass
                # raw_frame_queue.put({
                #     'frame': last_frame.copy(),
                #     'direction': last_direction
                # })
                data_params['flow_data']["frame"] = last_frame.copy()
                data_params['flow_data']["frame_time"] = time.time()
            print("Reintentando conexión en 5 segundos...")
            time.sleep(5)
    
    print("Hilo de captura terminado")

# ===== Hilo 2: Procesamiento con YOLO y Actuador =====
def processing_thread(data_params: dict):
    # Inicializar variables de seguimiento
    tracker_data = {
        'track_id': 1,
        'tracking_objects': {},
        'rod_count': 0,
        'counted_track_ids': set(),
        'center_points_prev_frame': []
    }
    
    # Variables para el actuador
    list_counter = []
    store_package = False
    actuactor_count = 0
    frame_count = 0
    prev_size = -1
    
    # Configuración de la cámara
    cam_params = CameraParameters(WIDTH, HEIGHT,
                                  x=data['x_init'], y=data['y_init'],
                                  w=data['roi_width'], h=data['roi_height'])
    cam_params.update_limits(data['counter_init'], data['counter_end'], data['counter_line'])

    # Cargar modelo YOLO
    model = YOLO(MODEL_PATH)
    print(f"Modelo YOLO cargado: {MODEL_PATH}")
    device = t_device("cuda" if t_cuda.is_available() else "cpu")
    model.to(device)
    print(f"Usando dispositivo: {device}")

    # Configurar logger y video writer
    storage_path = data['storage_path'] if data['storage_data'] else None
    logger = Logger(output_dir=data['logger_path'], storage_path=storage_path)
    video_writer = None
    
    if data['generate_video']:
        output_path = data['output_path']
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30.0
        frame_size = (data['roi_width'], data['roi_height'])
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        
        if not video_writer.isOpened():
            print(f"[ERROR] Could not initialize video writer for {output_path}")
            video_writer = None
        else:
            print(f"Video writer initialized: {output_path} at {fps} FPS, size {frame_size}")
    
    print("Hilo de procesamiento iniciado")
    last_time_processed = 0
    while not stop_event.is_set():
        try:
            # Obtener frame y dirección
            # data_received = raw_frame_queue.get(timeout=0.5)
            if last_time_processed >= data_params['flow_data']['frame_time'] or data_params['flow_data']['frame'] is None:
                time.sleep(0.0)
                continue
            
            full_frame = data_params['flow_data']['frame']
            direction = data_params['flow_data']['direction']
            direction_raw = data_params['flow_data']['direction_raw']
            direction_time = data_params['flow_data']['direction_time']
            serial_counter = data_params['flow_data']['serial_counter']
            print(f"direction: {direction}, direction_raw: {direction_raw}, time={direction_time}, serial_counter: {serial_counter}")
            # Recortar ROI
            roi_frame = full_frame[
                cam_params.y : cam_params.y + cam_params.h,
                cam_params.x : cam_params.x + cam_params.w
            ]
            
            # Realizar detecciones con YOLO
            detections = model(roi_frame, verbose=False, stream=False)

            # Procesar resultados
            if prev_size == len(list_counter):
                plot_historic(roi_frame, list_counter, data['logo'])

            center_points_cur_frame, actuator_pos = get_positions(
                detections,
                data['min_confidence'],
                data['actuator_data']
            )
            
            # Dibujar actuador si se detecta
            if actuator_pos[0] != 0 and actuator_pos[1] != 0:
                cv2.circle(roi_frame, (actuator_pos[0], actuator_pos[1]), 10, (0, 0, 255), -1)

            # Manejar lógica del actuador
            (list_counter,
             tracker_data,
             store_package,
             actuactor_count) = handle_actuator(
                 cam_params,
                 actuator_pos,
                 list_counter,
                 tracker_data,
                 store_package,
                 actuactor_count
             )
            
            # Procesar seguimiento solo si hay movimiento
            if direction != 0:
                # Ordenar puntos para seguimiento
                sorted_center_points = sorted(
                    center_points_cur_frame,
                    key=lambda point: point.pos_x
                )
                sorted_center_points.reverse()
                
                # Realizar seguimiento
                tracker = Tracker(
                    sorted_center_points,
                    roi_frame,
                    cam_params,
                    debug=data['debug'],
                    direction=direction
                )
                tracker.update_params(tracker_data)
                tracker_data = tracker.track()
                tracker.plot_count()
                
                # Resetear variables del actuador
                store_package = False
                actuactor_count = 0

            # Registrar frame si está habilitado el debug
            if data['debug']:
                logger.log(roi_frame, frame_count)
            
            frame_count += 1
            prev_size = len(list_counter)

            # Escribir en video si está habilitado
            if video_writer is not None:
                video_writer.write(roi_frame.copy())
            
            # Limpiar cola si está llena
            if processed_frame_queue.full():
                try:
                    processed_frame_queue.get_nowait()
                except queue.Empty:
                    pass
                    
            # Poner frame procesado en la cola
            processed_frame_queue.put(roi_frame)
            
        except queue.Empty:
            pass  # No hay frames disponibles, continuar
        except Exception as e:
            print(f"Error en procesamiento: {str(e)}")
            import traceback
            traceback.print_exc()

    # Liberar recursos al terminar
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
        threading.Thread(target=video_capture_thread, args=(data,), daemon=True),
        threading.Thread(target=processing_thread, args=(data,), daemon=True),
        threading.Thread(target=display_thread, daemon=True),
        threading.Thread(target=read_serial_thread, args=(data,), daemon=True),
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