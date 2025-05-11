#include <WiFi.h>
#include <WebSocketsServer.h>
#include <HNR-252_DCv0_1.h>
#include "esp32_cam_utils.h"

#define TRAINING_MODE true

// Настройка датчиков линии
const int IR_PINS[5] = {34, 35, 32, 33, 25};
const int IR_L = 32, IR_R = 33;
const int DIR_L = 4, PWM_L = 5, DIR_R = 18, PWM_R = 19;

// Параметры движения
const int BASE_SPEED = DEFAULT_BASE_SPEED;
const float KP = DEFAULT_KP;

// Глобальные объекты
MotorShield motors;
WebSocketsServer webSocket = WebSocketsServer(DEFAULT_WS_PORT);
uint8_t wsClientNum = 0xFF;
httpd_handle_t stream_httpd = NULL;

// WebSocket callback
void onWebSocketEvent(uint8_t num, WStype_t type, uint8_t * payload, size_t length) {
  if(type == WStype_CONNECTED) {
    wsClientNum = num;
    Serial.printf("[WS] Client #%u connected\n", num);
  }
  else if(type == WStype_DISCONNECTED && wsClientNum == num) {
    wsClientNum = 0xFF;
    Serial.printf("[WS] Client #%u disconnected\n", num);
  }
}

void setup() {
  Serial.begin(115200);
  
  // Инициализация камеры
  camera_config_t camera_config = get_default_camera_config();
  if (!init_camera(camera_config)) {
    return;
  }
  optimize_camera_settings();
  
  // Настройка Wi-Fi
  if (!setup_wifi(DEFAULT_WIFI_SSID, DEFAULT_WIFI_PASS)) {
    Serial.println("[WARN] WiFi connection failed, continuing without network");
  }
  
  // Запуск сервисов
  webSocket.begin();
  webSocket.onEvent(onWebSocketEvent);
  motors.setup();
  
  // Запуск HTTP сервера для стрима с камеры
  stream_httpd = start_mjpeg_server(DEFAULT_HTTP_PORT);
  
  Serial.println("[MODE] DATA COLLECTION with ESP32-CAM");
}

void loop() {
  static uint32_t lastTx = 0;
  webSocket.loop();
  
  float err = readLineError2(IR_L, IR_R);
  int steer = (int)(KP * err * 100);
  motors.runs((BASE_SPEED - steer)/2.5, (BASE_SPEED + steer)/2.5);
  
  if (millis() - lastTx >= 50) {
    transmitError(webSocket, wsClientNum, err);
    lastTx = millis();
  }
  
  Serial.printf("[ADC] L=%4d R=%4d  err=%.2f steer=%d\n", analogRead(IR_L), analogRead(IR_R), err, steer);
}