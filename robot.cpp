#include <WiFi.h>
#include <AsyncUDP.h>
#include <HNR-252_DCv0_1.h>

#define TRAINING_MODE false            // ← переключайте здесь

const char* WIFI_SSID = "KVANT_1511";
const char* WIFI_PASS = "7771111777";

#define USE_STA
IPAddress LAPTOP_IP(192,168,31,92);    // IP ноутбука
const uint16_t TX_PORT = 3333;         // ESP32 → ноут
const uint16_t RX_PORT = 2222;         // ноут  → ESP32

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
AsyncUDP udp;
volatile uint32_t lastCmdMs = 0;

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
  int8_t q = round(err * 127.0f);            // −127…+127
  udp.writeTo(&q,1,LAPTOP_IP,TX_PORT);
  Serial.printf("[TX] err=%+.2f  byte=%d\n", err, q);
}

/* ───────────────── setup ─────────────────────────────*/
void setup() {
  Serial.begin(115200);

  // Wi-Fi STA
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("[WiFi] Connecting");
  while (WiFi.status()!=WL_CONNECTED){ delay(500); Serial.print("."); }
  Serial.printf("\n[WiFi] Connected  IP=%s\n",
                WiFi.localIP().toString().c_str());

  // UDP
  if(!udp.listen(RX_PORT)) Serial.println("[UDP] listen failed!");
  udp.onPacket([](AsyncUDPPacket p){
    if(!TRAINING_MODE && p.length()==1){
      int8_t steer = *p.data();              // −127…+127
      int left  = BASE_SPEED - steer;
      int right = BASE_SPEED + steer;
      motors.runs(left/2.5, right/2.5);
      lastCmdMs = millis();
      Serial.printf("[RX] steer=%d  L=%d  R=%d\n", steer, left, right);
    }
  });

  motors.setup();
  Serial.println(TRAINING_MODE? "[MODE] TRAINING" : "[MODE] INFERENCE");
}

/* ───────────────── loop ──────────────────────────────*/
void loop() {
  static uint32_t lastTx = 0;

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



