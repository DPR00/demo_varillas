import cv2

url = "rtsp://admin:Hik12345@192.168.18.5/Streaming/Channels/101"

cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("No se pudo abrir el stream RTSP.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer frame.")
        break

    cv2.imshow("RTSP", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
