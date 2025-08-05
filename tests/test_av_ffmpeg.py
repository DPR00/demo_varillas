# LA LATENCIA ES MÍNIMA, CASI IMPERCEPTIBLE PERO EL PRINCIPAL PROBLEMA ES QUE SE PIERDE EL VIDEO
# DE FORMA ALEATORIA, NO SE POR QUÉ, PERO EL VIDEO SE PIERDE Y SE CONGELA
import av
import cv2
import numpy as np
import time
from av.error import FFmpegError

# Configuración de la cámara
# RTSP_URL = "rtsp://admin:IA+T3cCAM@192.168.18.3/Streaming/Channels/101"
# RTSP_URL = "rtsp://admin:IA+T3cCAM@192.168.18.3/Streaming/Channels/101?transportmode=unicast"
RTSP_URL = "media/test_30_07_25_17_18pm.mp4"

# Opciones optimizadas para conexiones inestables
FFMPEG_OPTIONS = {
    'rtsp_transport': 'tcp',           # Transporte confiable
    'fflags': 'nobuffer',
    'flags': 'low_delay',
    'max_delay': '500000',             # 500 ms
    'analyzeduration': '100000',
    'probesize': '1024',
    'tune': 'zerolatency',
    'framedrop': '1',
    'avioflags': 'direct',
    'flush_packets': '1',
    'timeout': '5000000',              # 5 segundos
    'reconnect': '1',                  # Reconectar automáticamente
    'reconnect_at_eof': '1',
    'reconnect_streamed': '1',
    'reconnect_delay_max': '5',        # Máximo 5 segundos entre reconexiones
    'stimeout': '5000000',             # Timeout de socket
    'heartbeat_interval': '10'
}

def main():
    last_frame = None
    
    while True:
        try:
            print("Conectando a la cámara...")
            container = av.open(RTSP_URL, options=FFMPEG_OPTIONS)
            stream = container.streams.video[0]
            
            print("Conexión establecida. Mostrando video...")
            for packet in container.demux(stream):
                for frame in packet.decode():
                    # Convertir frame a formato OpenCV
                    img = frame.to_ndarray(format='bgr24')
                    last_frame = img
                    cv2.imshow('Hikvision Low-Latency', img)
                    
                    # Salir con 'q'
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        container.close()
                        cv2.destroyAllWindows()
                        return
                        
        except FFmpegError as e:
            print(f"Error de conexión: {e}")
            print("Intentando reconectar en 2 segundos...")
            
            # Mostrar último frame durante la reconexión
            if last_frame is not None:
                cv2.putText(last_frame, "RECONECTANDO...", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Hikvision Low-Latency', last_frame)
                cv2.waitKey(1)
            
            time.sleep(2)
            
        except KeyboardInterrupt:
            print("Programa detenido por el usuario")
            if 'container' in locals():
                container.close()
            cv2.destroyAllWindows()
            return
            
        except Exception as e:
            print(f"Error inesperado: {e}")
            print("Reintentando en 5 segundos...")
            time.sleep(5)

if __name__ == "__main__":
    main()