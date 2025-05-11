#include "esp32_cam_utils.h"

#define TRAINING_MODE false

// Пины для моторов
const int DIR_L = 4, PWM_L = 5, DIR_R = 18, PWM_R = 19;
const int BASE_SPEED = DEFAULT_BASE_SPEED;
const uint32_t CMD_TIMEOUT = DEFAULT_CMD_TIMEOUT;

// Глобальные объекты
MotorShield motors;
WebSocketsServer webSocket = WebSocketsServer(DEFAULT_WS_PORT);
volatile uint32_t lastCmdMs = 0;
uint8_t wsClientNum = 0xFF;
httpd_handle_t stream_httpd = NULL;

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
  
  // Инициализация камеры
  camera_config_t camera_config = get_default_camera_config();
  if (!init_camera(camera_config)) {
    return;
  }
  optimize_camera_settings();
  
  // Настройка Wi-Fi
  if (!setup_wifi(DEFAULT_WIFI_SSID, DEFAULT_WIFI_PASS)) {
    Serial.println("[WARN] WiFi connection failed, continuing without network");
    return; // В режиме удаленного управления без WiFi работать не можем
  }
  
  webSocket.begin();
  webSocket.onEvent(onWebSocketEvent);
  motors.setup();
  
  // Запуск HTTP сервера для стрима с камеры
  stream_httpd = start_mjpeg_server(DEFAULT_HTTP_PORT);
  
  Serial.println("[MODE] REMOTE CONTROL with ESP32-CAM");
}

void loop() {
  webSocket.loop();
  if (millis() - lastCmdMs > CMD_TIMEOUT) {
    motors.brake();
    Serial.println("[TIMEOUT] steer not received -> BRAKE");
    lastCmdMs = millis() + 0x7FFFFFFF/2;  // Чтобы не спамить сообщениями
  }
}