#include <WiFi.h>
#include <WebSocketsServer.h>
#include <HNR-252_DCv0_1.h>

#define TRAINING_MODE false

const char* WIFI_SSID = "KVANT_1511";
const char* WIFI_PASS = "7771111777";
#define USE_STA
const uint16_t WS_PORT = 2222;

const int DIR_L = 4, PWM_L = 5, DIR_R = 18, PWM_R = 19;
const int BASE_SPEED = 120;
const uint32_t CMD_TIMEOUT = 250;
MotorShield motors;
WebSocketsServer webSocket = WebSocketsServer(WS_PORT);
volatile uint32_t lastCmdMs = 0;
uint8_t wsClientNum = 0xFF;

void onWebSocketEvent(uint8_t num, WStype_t type, uint8_t * payload, size_t length) {
  if(type == WStype_CONNECTED) {
    wsClientNum = num;
    Serial.printf("[WS] Client #%u connected\n", num);
  }
  else if(type == WStype_DISCONNECTED && wsClientNum == num) {
    wsClientNum = 0xFF;
    motors.brake();
    Serial.printf("[WS] Client #%u disconnected, motors stopped\n", num);
  }
  else if(type == WStype_BIN && length == 1) {
    int8_t steer = (int8_t)payload[0];
    int left  = BASE_SPEED - steer;
    int right = BASE_SPEED + steer;
    motors.runs(left/2.5, right/2.5);
    lastCmdMs = millis();
    Serial.printf("[WS-RX] steer=%d  L=%d  R=%d\n", steer, left, right);
  }
}

void setup() {
  Serial.begin(115200);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("[WiFi] Connecting");
  while (WiFi.status()!=WL_CONNECTED){ delay(500); Serial.print("."); }
  Serial.printf("\n[WiFi] Connected  IP=%s\n", WiFi.localIP().toString().c_str());
  webSocket.begin();
  webSocket.onEvent(onWebSocketEvent);
  motors.setup();
  Serial.println("[MODE] REMOTE CONTROL");
}

void loop() {
  webSocket.loop();
  if (millis() - lastCmdMs > CMD_TIMEOUT) {
    motors.brake();
    Serial.println("[TIMEOUT] steer not received -> BRAKE");
    lastCmdMs = millis() + 0x7FFFFFFF/2;
  }
} 