import cv2
import threading
import queue
import time
import numpy as np
from ultralytics import YOLO  # pip install ultralytics

# ===== Configuración Global =====
RTSP_URL = "rtsp://admin:IA+T3cCAM@192.168.18.3/Streaming/Channels/101?transportmode=unicast"
# RTSP_URL = "media/test_30_07_25_17_18pm.mp4"  # O tu URL RTSP
MODEL_PATH = "yolov8n.pt"  # Modelo YOLOv8 preentrenado
BUFFER_SIZE = 1  # Tamaño del buffer para baja latencia

# Colas para comunicación entre hilos
raw_frame_queue = queue.Queue(maxsize=2)       # Frames sin procesar
processed_frame_queue = queue.Queue(maxsize=2)  # Frames procesados con detecciones
stop_event = threading.Event()                 # Señal de parada para todos los hilos

# ===== Hilo 1: Captura de Video =====
def video_capture_thread():
    cap = cv2.VideoCapture(RTSP_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, BUFFER_SIZE)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
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
            continue
        
        # Limpiar cola si está llena para mantener solo el frame más reciente
        if raw_frame_queue.full():
            try:
                raw_frame_queue.get_nowait()
            except queue.Empty:
                pass
        
        raw_frame_queue.put(frame)
    
    cap.release()
    print("Hilo de captura terminado")

# ===== Hilo 2: Procesamiento con YOLO =====
def processing_thread():
    # Cargar modelo YOLO
    model = YOLO(MODEL_PATH)
    print(f"Modelo YOLO cargado: {MODEL_PATH}")
    
    # Configurar aceleración (si está disponible)
    try:
        model.to('cuda')  # Usar GPU si está disponible
        print("Usando aceleración GPU")
    except:
        print("Usando CPU")
    
    print("Hilo de procesamiento iniciado")
    while not stop_event.is_set():
        try:
            # Obtener el último frame disponible (esperar máximo 0.5s)
            frame = raw_frame_queue.get(timeout=0.5)
            
            # Realizar detecciones con YOLO
            results = model(frame, verbose=False, stream=False)
            
            # Procesar resultados
            processed_frame = frame.copy()
            for result in results:
                # Dibujar cajas y etiquetas
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0].item()
                    cls_id = int(box.cls[0])
                    label = f"{result.names[cls_id]} {conf:.2f}"
                    
                    # Dibujar caja
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(processed_frame, label, (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Limpiar cola si está llena para mantener solo el frame más reciente
            if processed_frame_queue.full():
                try:
                    processed_frame_queue.get_nowait()
                except queue.Empty:
                    pass
            
            processed_frame_queue.put(processed_frame)
            
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
            display_frame = cv2.resize(processed_frame, (1280, 720))
            
            # Mostrar FPS y estado
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Mostrar tamaño de colas
            cv2.putText(display_frame, f"Captura: {raw_frame_queue.qsize()}/2", (20, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Procesados: {processed_frame_queue.qsize()}/2", (20, 100), 
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