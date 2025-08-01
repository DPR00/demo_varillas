#include <HardwareSerial.h>

// Definición de pines (ajusta según tu hardware)
// const int inputPins[4] = { 12, 14, 27, 26 };
const int inputPins[2] = { 27, 26 };

void setup() {
  // Inicializa comunicación serial a 115200 baudios
  Serial.begin(115200);
  while (!Serial) {
    ; // Espera que Serial esté listo
  }

  // Configura cada pin como entrada (o INPUT_PULLUP si usas pull-up interno)
  for (int i = 0; i < 2; i++) {
    pinMode(inputPins[i], INPUT_PULLUP);
  }
}

void loop() {
  // Leer cada entrada y enviar siempre el estado como una cadena de bits "1010"
  String bitString;
  bitString.reserve(2);
  for (int i = 0; i < 2; i++) {
    bitString += digitalRead(inputPins[i]);
  }

  Serial.println(bitString);
  delay(100); // Ajusta la frecuencia de envío (100 ms aquí)
}