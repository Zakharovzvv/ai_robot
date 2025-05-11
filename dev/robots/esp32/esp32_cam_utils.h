#ifndef ESP32_CAM_UTILS_H
#define ESP32_CAM_UTILS_H

#include <WiFi.h>
#include <WebSocketsServer.h>
#include <HNR-252_DCv0_1.h>
#include "esp_camera.h"
#include "esp_http_server.h"

// Константы WiFi
const char* DEFAULT_WIFI_SSID = "KVANT_1511";
const char* DEFAULT_WIFI_PASS = "7771111777";
const uint16_t DEFAULT_WS_PORT = 2222;
const uint16_t DEFAULT_HTTP_PORT = 81;

// Константы для мотора и датчиков
const int DEFAULT_BASE_SPEED = 120;
const float DEFAULT_KP = 3.0f;
const uint32_t DEFAULT_CMD_TIMEOUT = 250; // мс

// Пины для ESP32-CAM (AI Thinker модель)
#define CAM_PIN_PWDN    32
#define CAM_PIN_RESET   -1
#define CAM_PIN_XCLK     0
#define CAM_PIN_SIOD    26
#define CAM_PIN_SIOC    27
#define CAM_PIN_D7      35
#define CAM_PIN_D6      34
#define CAM_PIN_D5      39
#define CAM_PIN_D4      36
#define CAM_PIN_D3      21
#define CAM_PIN_D2      19
#define CAM_PIN_D1      18
#define CAM_PIN_D0       5
#define CAM_PIN_VSYNC   25
#define CAM_PIN_HREF    23
#define CAM_PIN_PCLK    22

// Конфигурация камеры по умолчанию
inline camera_config_t get_default_camera_config() {
  camera_config_t config;
  config.pin_pwdn = CAM_PIN_PWDN;
  config.pin_reset = CAM_PIN_RESET;
  config.pin_xclk = CAM_PIN_XCLK;
  config.pin_sccb_sda = CAM_PIN_SIOD;
  config.pin_sccb_scl = CAM_PIN_SIOC;
  config.pin_d7 = CAM_PIN_D7;
  config.pin_d6 = CAM_PIN_D6;
  config.pin_d5 = CAM_PIN_D5;
  config.pin_d4 = CAM_PIN_D4;
  config.pin_d3 = CAM_PIN_D3;
  config.pin_d2 = CAM_PIN_D2;
  config.pin_d1 = CAM_PIN_D1;
  config.pin_d0 = CAM_PIN_D0;
  config.pin_vsync = CAM_PIN_VSYNC;
  config.pin_href = CAM_PIN_HREF;
  config.pin_pclk = CAM_PIN_PCLK;
  config.xclk_freq_hz = 20000000;
  config.ledc_timer = LEDC_TIMER_0;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_SVGA;    // 800x600
  config.jpeg_quality = 12;              // 0-63, чем меньше число тем лучше качество
  config.fb_count = 2;                   // Кол-во буферов кадров
  config.grab_mode = CAMERA_GRAB_LATEST; // Всегда берем последний кадр
  return config;
}

// Оптимизация камеры для слабого освещения
inline void optimize_camera_settings() {
  sensor_t *s = esp_camera_sensor_get();
  s->set_brightness(s, 1);   // -2 to 2
  s->set_contrast(s, 1);     // -2 to 2
  s->set_saturation(s, 0);   // -2 to 2
  s->set_special_effect(s, 0); // 0 - No Effect
  s->set_whitebal(s, 1);     // 0 = disable, 1 = enable
  s->set_awb_gain(s, 1);     // 0 = disable, 1 = enable
  s->set_wb_mode(s, 0);      // 0 to 4 - auto, sunny, cloudy, home, office 
  s->set_exposure_ctrl(s, 1);// 0 = disable, 1 = enable
  s->set_gain_ctrl(s, 1);    // 0 = disable, 1 = enable
  s->set_agc_gain(s, 12);    // 0 to 30
  s->set_gainceiling(s, (gainceiling_t)1);  // 0 to 6
  s->set_raw_gma(s, 1);      // 0 = disable, 1 = enable
  s->set_hmirror(s, 0);      // 0 = disable, 1 = enable
  s->set_vflip(s, 0);        // 0 = disable, 1 = enable
}

// Вариант с 2-датчиками: −1…+1
inline float readLineError2(int IR_L, int IR_R) {
  int rawL = analogRead(IR_L);
  int rawR = analogRead(IR_R);
  float nL = 1.0f - rawL/1000.0f;
  float nR = 1.0f - rawR/1000.0f;
  float sum = nL + nR;
  if(sum < 0.05f) return 0.0f;  // линия потеряна
  return (nR - nL) / sum;       // −1…+1
}

// Вариант с 5-датчиками: −2…+2
inline float readLineError5(const int* IR_PINS) {
  long num = 0, den = 0;
  for (int i = 0; i < 5; ++i) {
    int v = 4095 - analogRead(IR_PINS[i]);   // инверсия
    num += v * (i - 2);                      // -2…+2
    den += v;
  }
  if (den < 2000) return 0;
  return (float)num / den;                   // −2…+2
}

// Отправить значение ошибки через WebSocket
inline void transmitError(WebSocketsServer &webSocket, uint8_t wsClientNum, float err) {
  if(wsClientNum != 0xFF) {
    int8_t q = round(err * 127.0f);            // −127…+127
    uint8_t buf[1] = { (uint8_t)q };
    webSocket.sendBIN(wsClientNum, buf, 1);
  }
}

// MJPEG стрим обработчик
static esp_err_t mjpeg_stream_handler(httpd_req_t *req) {
  camera_fb_t * fb = NULL;
  esp_err_t res = ESP_OK;
  size_t _jpg_buf_len = 0;
  uint8_t * _jpg_buf = NULL;
  char * part_buf[64];
  
  static int64_t last_frame = 0;
  if(!last_frame) {
    last_frame = esp_timer_get_time();
  }

  res = httpd_resp_set_type(req, "multipart/x-mixed-replace;boundary=123456789000000000000987654321");
  if(res != ESP_OK){
    return res;
  }

  while(true){
    fb = esp_camera_fb_get();
    if(!fb) {
      Serial.println("Camera capture failed");
      res = ESP_FAIL;
      break;
    }
    
    // Используем уже готовый JPEG буфер или конвертируем в JPEG при необходимости
    if(fb->format != PIXFORMAT_JPEG){
      bool jpeg_converted = frame2jpg(fb, 80, &_jpg_buf, &_jpg_buf_len);
      esp_camera_fb_return(fb);
      fb = NULL;
      if(!jpeg_converted){
        Serial.println("JPEG compression failed");
        res = ESP_FAIL;
        break;
      }
    } else {
      _jpg_buf_len = fb->len;
      _jpg_buf = fb->buf;
    }

    // Отправляем часть multipart/x-mixed-replace
    if(res == ESP_OK){
      size_t hlen = snprintf((char *)part_buf, 64, 
        "--123456789000000000000987654321\r\n"
        "Content-Type: image/jpeg\r\n"
        "Content-Length: %u\r\n\r\n", 
        _jpg_buf_len);
      res = httpd_resp_send_chunk(req, (const char *)part_buf, hlen);
    }
    
    // Отправляем данные JPEG изображения
    if(res == ESP_OK){
      res = httpd_resp_send_chunk(req, (const char *)_jpg_buf, _jpg_buf_len);
    }
    
    // Завершаем текущий фрейм
    if(res == ESP_OK){
      res = httpd_resp_send_chunk(req, "\r\n", 2);
    }
    
    // Если буфер был выделен для конвертации, освобождаем его
    if(fb){
      esp_camera_fb_return(fb);
      fb = NULL;
      _jpg_buf = NULL;
    } else if(_jpg_buf){
      free(_jpg_buf);
      _jpg_buf = NULL;
    }
    
    // Проверяем подключение клиента
    if(res != ESP_OK){
      break;
    }
    
    // Регулируем FPS (необязательно)
    int64_t fr_end = esp_timer_get_time();
    int64_t frame_time = fr_end - last_frame;
    last_frame = fr_end;
    frame_time /= 1000; // микросек -> мсек
    Serial.printf("[MJPEG] %uKB %ums (%.1ffps)\n", 
      (uint32_t)(_jpg_buf_len/1024), 
      (uint32_t)frame_time, 1000.0 / (uint32_t)frame_time);
      
    // Небольшая задержка чтобы не перегружать процессор
    delay(20);
  }

  last_frame = 0;
  return res;
}

// Настройка HTTP сервера для стриминга
inline httpd_handle_t start_mjpeg_server(int port = DEFAULT_HTTP_PORT) {
  httpd_handle_t stream_httpd = NULL;
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = port;
  config.ctrl_port = port;
  config.max_uri_handlers = 2;
  
  httpd_uri_t stream_uri = {
    .uri       = "/stream",
    .method    = HTTP_GET,
    .handler   = mjpeg_stream_handler,
    .user_ctx  = NULL
  };

  // Запускаем HTTP сервер
  Serial.printf("[INFO] Starting web server on port: %d\n", port);
  if (httpd_start(&stream_httpd, &config) == ESP_OK) {
    // Регистрируем обработчик для стрима
    httpd_register_uri_handler(stream_httpd, &stream_uri);
    Serial.print("[INFO] Camera stream available at http://");
    Serial.print(WiFi.localIP());
    Serial.printf(":%d/stream\n", port);
  }
  
  return stream_httpd;
}

// Настройка подключения к WiFi
inline bool setup_wifi(const char* ssid, const char* password, int max_attempts = 20) {
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  Serial.print("[WiFi] Connecting");
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < max_attempts) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\n[WiFi] Connected  IP=%s\n", WiFi.localIP().toString().c_str());
    return true;
  } else {
    Serial.println("\n[WiFi] Failed to connect");
    return false;
  }
}

// Инициализация ESP32-CAM
inline bool init_camera(camera_config_t &config) {
  Serial.println("[INFO] Initializing camera...");
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("[ERROR] Camera init failed with error 0x%x\n", err);
    return false;
  }
  Serial.println("[INFO] Camera initialized successfully");
  return true;
}

#endif // ESP32_CAM_UTILS_H