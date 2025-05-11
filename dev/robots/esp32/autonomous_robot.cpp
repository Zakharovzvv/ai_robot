#include <WiFi.h>
#include <HNR-252_DCv0_1.h>
#include "esp32_cam_utils.h"
#include "model_data.cc"

// TFLite Micro
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Настройка Wi-Fi для отладки (опционально)
bool wifi_enabled = true;      // Установите в false для полностью автономного режима

// Параметры модели
constexpr int IMG_W = 160;
constexpr int IMG_H = 120;
constexpr int IMG_C = 3;
constexpr int kTensorArenaSize = 60 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// Пины и моторы
const int DIR_L = 4, PWM_L = 5, DIR_R = 18, PWM_R = 19;
const int BASE_SPEED = DEFAULT_BASE_SPEED;
MotorShield motors;
httpd_handle_t stream_httpd = NULL;

// TFLite Micro объекты
tflite::MicroErrorReporter micro_error_reporter;
tflite::AllOpsResolver resolver;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Последнее значение смещения от нейросети для отображения в MJPEG стриме
float last_offset = 0.0;

void setup() {
  Serial.begin(115200);
  motors.setup();

  // Инициализация камеры
  camera_config_t camera_config = get_default_camera_config();
  // Для нейросети нам нужен формат RGB888
  camera_config.pixel_format = PIXFORMAT_RGB888;
  camera_config.frame_size = FRAMESIZE_QVGA; // 320x240, будем ресайзить
  
  if (!init_camera(camera_config)) {
    return;
  }
  optimize_camera_settings();

  // Настройка Wi-Fi, если включен
  if (wifi_enabled) {
    if (setup_wifi(DEFAULT_WIFI_SSID, DEFAULT_WIFI_PASS, 20)) {
      // Запуск HTTP сервера для стрима с камеры
      stream_httpd = start_mjpeg_server(DEFAULT_HTTP_PORT);
    } else {
      Serial.println("[WiFi] Running in offline mode");
    }
  }

  // Загрузка модели TFLite
  Serial.println("[NN] Loading TFLite model...");
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("[ERROR] Model schema version mismatch!");
    while (1);
  }
  
  interpreter = new tflite::MicroInterpreter(model, resolver, tensor_arena, kTensorArenaSize, &micro_error_reporter);
  interpreter->AllocateTensors();
  input = interpreter->input(0);
  output = interpreter->output(0);
  Serial.println("[NN] Model loaded and ready for inference!");
  
  // Проверяем параметры модели
  Serial.printf("[NN] Model input shape: %dx%dx%d\n", IMG_W, IMG_H, IMG_C);
  Serial.printf("[NN] TensorArena size: %d bytes\n", kTensorArenaSize);
  
  // Пауза перед запуском основного цикла
  delay(500);
  Serial.println("[SYSTEM] Autonomous robot starting...");
}

void loop() {
  // 1. Снять кадр с камеры
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed");
    delay(100);
    return;
  }

  // 2. Преобразование изображения для нейросети: ресайз и нормализация
  for (int c = 0; c < IMG_C; ++c) {
    for (int y = 0; y < IMG_H; ++y) {
      for (int x = 0; x < IMG_W; ++x) {
        int src_x = x * fb->width / IMG_W;
        int src_y = y * fb->height / IMG_H;
        int src_idx = (src_y * fb->width + src_x) * 3 + c;
        
        // Централизованная нормализация (-1 до +1) вместо [0, 1]
        if (src_idx < fb->len) {
          input->data.f[c * IMG_H * IMG_W + y * IMG_W + x] = 
            (fb->buf[src_idx] / 255.0f - 0.5f) / 0.5f;
        }
      }
    }
  }
  esp_camera_fb_return(fb);

  // 3. Инференс
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("Invoke failed!");
    delay(100);
    return;
  }
  float offset = output->data.f[0];
  last_offset = offset; // Сохраняем для отображения в MJPEG стриме

  // 4. Управление моторами
  int steer = int(offset * 127.0f);
  int left  = BASE_SPEED - steer;
  int right = BASE_SPEED + steer;
  motors.runs(left/2.5, right/2.5);

  // 5. Отладка
  Serial.printf("[NN] offset=%.3f steer=%d L=%d R=%d\n", offset, steer, left, right);

  delay(20); // ~50 Гц
}