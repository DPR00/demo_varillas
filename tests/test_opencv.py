import cv2
import os

# Configuración de la cámara (reemplaza con tus datos)
RTSP_URL = "rtsp://admin:IA+T3cCAM@192.168.18.3/Streaming/Channels/101?transportmode=unicast"
# RTSP_URL = "media/operation_1920x1080.mp4"
BUFFER_SIZE = 1  # Reduce el buffer para menor latencia

# Usar decodificación por hardware (si tienes GPU NVIDIA)
# os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "hwaccel;cuda"  # NVIDIA

# Inicializar capturador de video con configuración de baja latencia
cap = cv2.VideoCapture(RTSP_URL)
# cap.set(cv2.CAP_PROP_BUFFERSIZE, BUFFER_SIZE)

# Configuración adicional para baja latencia (requiere OpenCV 4+)
# cap.set(cv2.CAP_PROP_FPS, 30)  # FPS de tu cámara
# cap.set(cv2.CAP_PROP_FPS, 30)  # FPS de tu cámara
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H','2','6','4'))  # Códec H.264
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))  # Alternativa: MJPEG
tries = 0
while True:
    success, frame = cap.read()
    if not success:
        print("Error de conexión")
        tries += 1
        if tries>3: break
        continue
    
    # Mostrar frame (reducir tamaño para mejor rendimiento)
    display_frame = cv2.resize(frame, (1280, 720))  # Ajustar según necesidad
    cv2.imshow("Hikvision Live", display_frame)
    
    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()