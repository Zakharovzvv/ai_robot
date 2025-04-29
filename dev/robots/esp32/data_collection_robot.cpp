#include <WiFi.h>
#include <WebSocketsServer.h>
#include <HNR-252_DCv0_1.h>

#define TRAINING_MODE true

const char* WIFI_SSID = "KVANT_1511";
const char* WIFI_PASS = "7771111777";
#define USE_STA
const uint16_t WS_PORT = 2222;

const int DIR_L = 4, PWM_L = 5, DIR_R = 18, PWM_R = 19;
const int IR_PINS[5] = {34, 35, 32, 33, 25};
const int IR_L = 32, IR_R = 33;
const int BASE_SPEED = 120;
const float KP = 3.0f;
MotorShield motors;
WebSocketsServer webSocket = WebSocketsServer(WS_PORT);
uint8_t wsClientNum = 0xFF;

float readLineError2() {
  int rawL = analogRead(IR_L);
  int rawR = analogRead(IR_R);
  float nL = 1.0f - rawL/1000.0f;
  float nR = 1.0f - rawR/1000.0f;
  float sum = nL + nR;
  if(sum < 0.05f) return 0.0f;
  return (nR - nL) / sum;
}

void transmitError(float err){
  if(wsClientNum != 0xFF) {
    int8_t q = round(err * 127.0f);
    uint8_t buf[1] = { (uint8_t)q };
    webSocket.sendBIN(wsClientNum, buf, 1);
  }
}

void onWebSocketEvent(uint8_t num, WStype_t type, uint8_t * payload, size_t length) {
  if(type == WStype_CONNECTED) wsClientNum = num;
  else if(type == WStype_DISCONNECTED && wsClientNum == num) wsClientNum = 0xFF;
}

void setup() {
  Serial.begin(115200);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status()!=WL_CONNECTED){ delay(500); }
  webSocket.begin();
  webSocket.onEvent(onWebSocketEvent);
  motors.setup();
  Serial.println("[MODE] DATA COLLECTION");
}

void loop() {
  static uint32_t lastTx = 0;
  webSocket.loop();
  float err = readLineError2();
  int steer = (int)(KP * err * 100);
  motors.runs((BASE_SPEED - steer)/2.5, (BASE_SPEED + steer)/2.5);
  if (millis() - lastTx >= 50) {
    transmitError(err);
    lastTx = millis();
  }
  Serial.printf("[ADC] L=%4d R=%4d  err=%.2f steer=%d\n", analogRead(IR_L), analogRead(IR_R), err, steer);
} 