#include <WiFi.h>
#include <WebSocketsServer.h>
#include <HNR-252_DCv0_1.h>

#define TRAINING_MODE true

const char* WIFI_SSID = "KVANT_1511";
const char* WIFI_PASS = "7771111777";

#define USE_STA
const uint16_t WS_PORT = 2222;         // WebSocket порт

// ── ваши пины ──────────────────────────────
const int DIR_L = 4;
const int PWM_L = 5;
const int DIR_R = 18;
const int PWM_R = 19;

// 5-канальная линейка (оставил как есть)
const int IR_PINS[5] = {34, 35, 32, 33, 25}; // слева → справа
// два датчика (L,R) – используем их же
const int IR_L = 32;
const int IR_R = 33;

// ── параметры движения ─────────────────────
const int   BASE_SPEED = 120;      // 0-255
const float KP         = 3.0f;     // при TRAINING_MODE
const uint32_t CMD_TIMEOUT = 250;  // мс

MotorShield motors;
WebSocketsServer webSocket = WebSocketsServer(WS_PORT);
volatile uint32_t lastCmdMs = 0;
uint8_t wsClientNum = 0xFF; // номер подключенного клиента

/* ───────────────── helpers ───────────────────────────*/
// Вариант с 2-датчиками: −1…+1
float readLineError2() {
  int rawL = analogRead(IR_L);
  int rawR = analogRead(IR_R);
  float nL = 1.0f - rawL/1000.0f;
  float nR = 1.0f - rawR/1000.0f;
  float sum = nL + nR;
  if(sum < 0.05f) return 0.0f;            // линия потеряна
  return (nR - nL) / sum;                 // −1…+1
}

// Старый вариант на 5-датчиков (оставлен нетронутым)
float readLineError5() {
  long num = 0, den = 0;
  for (int i = 0; i < 5; ++i) {
    int v = 4095 - analogRead(IR_PINS[i]);   // инверсия
    num += v * (i - 2);                      // -2…+2
    den += v;
  }
  if (den < 2000) return 0;
  return (float)num / den;                   // −2…+2
}

void transmitError(float err){
  if(wsClientNum != 0xFF) {
    int8_t q = round(err * 127.0f);            // −127…+127
    uint8_t buf[1] = { (uint8_t)q };
    webSocket.sendBIN(wsClientNum, buf, 1);
    Serial.printf("[WS-TX] err=%+.2f  byte=%d\n", err, q);
  }
}

/* ───────────────── WebSocket callback ────────────────*/
void onWebSocketEvent(uint8_t num, WStype_t type, uint8_t * payload, size_t length) {
  if(type == WStype_CONNECTED) {
    wsClientNum = num;
    Serial.printf("[WS] Client #%u connected\n", num);
  }
  else if(type == WStype_DISCONNECTED) {
    Serial.printf("[WS] Client #%u disconnected\n", num);
    if(wsClientNum == num) wsClientNum = 0xFF;
    motors.brake();
  }
  else if(type == WStype_BIN && !TRAINING_MODE && length == 1) {
    int8_t steer = (int8_t)payload[0];              // −127…+127
    int left  = BASE_SPEED - steer;
    int right = BASE_SPEED + steer;
    motors.runs(left/2.5, right/2.5);
    lastCmdMs = millis();
    Serial.printf("[WS-RX] steer=%d  L=%d  R=%d\n", steer, left, right);
  }
}

/* ───────────────── setup ─────────────────────────────*/
void setup() {
  Serial.begin(115200);

  // Wi-Fi STA
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("[WiFi] Connecting");
  while (WiFi.status()!=WL_CONNECTED){ delay(500); Serial.print("."); }
  Serial.printf("\n[WiFi] Connected  IP=%s\n", WiFi.localIP().toString().c_str());

  // WebSocket
  webSocket.begin();
  webSocket.onEvent(onWebSocketEvent);

  motors.setup();
  Serial.println("[MODE] DATA COLLECTION");
}

/* ───────────────── loop ──────────────────────────────*/
void loop() {
  static uint32_t lastTx = 0;
  webSocket.loop();

  if (TRAINING_MODE) {
    /* 1. Читаем линию двумя датчиками */
    float err = readLineError2();            // −1…+1
    int   steer = (int)(KP * err * 100);     // как у вас было
    motors.runs((BASE_SPEED - steer)/2.5,
                (BASE_SPEED + steer)/2.5);

    /* 2. Лог-АЦП (можно закомментировать) */
    Serial.printf("[ADC] L=%4d R=%4d  err=%.2f steer=%d\n",
                  analogRead(IR_L), analogRead(IR_R), err, steer);

    /* 3. Шлём ошибку раз в 50 мс */
    if (millis() - lastTx >= 50) {
      transmitError(err);                    // −1…+1  → byte
      lastTx = millis();
    }
  }
  else { // INFERENCE MODE
    if (millis() - lastCmdMs > CMD_TIMEOUT) {
      motors.brake();
      Serial.println("[TIMEOUT] steer not received -> BRAKE");
      lastCmdMs = millis() + 0x7FFFFFFF/2;   // чтобы не спамить
    }
  }
}



