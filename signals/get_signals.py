import serial
import time

# const int inputPins[4] = { 12, 14, 27, 26 };
# const int inputPins[4] = {  27, 26 }; RETROCEDE, AVANZA

# Ajusta el puerto y la velocidad según tu configuración\ nSERIAL_PORT = 'COM3'  # Windows: 'COM3', Linux: '/dev/ttyUSB0'
SERIAL_PORT = 'COM3'  # Linux: '/dev/ttyUSB0', Windows: 'COM3'
BAUD_RATE = 115200
TIMEOUT = 1  # segundos


def main():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)
    except serial.SerialException as e:
        print(f"Error al abrir {SERIAL_PORT}: {e}")
        return

    # Dar tiempo para que el ESP32 reinicie y desechar logs de arranque
    time.sleep(2)
    ser.reset_input_buffer()

    print(f"Leyendo bits desde {SERIAL_PORT} a {BAUD_RATE} baudios...")
    try:
        while True:
            raw = ser.readline()
            if not raw:
                continue
            try:
                line = raw.decode('utf-8').strip()
            except UnicodeDecodeError:
                continue

            # Si la línea consiste en exactamente 4 caracteres de '0' o '1', la mostramos
            if len(line) == 2 and all(c in '01' for c in line):
                print(line)
            # En otro caso, simplemente ignoramos la línea

    except KeyboardInterrupt:
        print("Deteniendo lectura...")
    finally:
        ser.close()

if __name__ == '__main__':
    main()