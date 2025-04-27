#include <WiFi.h>
#include <HNR-252_DCv0_1.h>
#include "model_data.cc"

// TFLite Micro
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Параметры модели
constexpr int IMG_W = 160;
constexpr int IMG_H = 120;
constexpr int IMG_C = 3;
constexpr int kTensorArenaSize = 60 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// Пины и моторы
const int DIR_L = 4, PWM_L = 5, DIR_R = 18, PWM_R = 19;
const int BASE_SPEED = 120;
MotorShield motors;

// Камера (пример для OV2640, настройте под свою!)
#include "esp_camera.h"
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

camera_config_t camera_config = {
  .pin_pwdn  = CAM_PIN_PWDN,
  .pin_reset = CAM_PIN_RESET,
  .pin_xclk = CAM_PIN_XCLK,
  .pin_sccb_sda = CAM_PIN_SIOD,
  .pin_sccb_scl = CAM_PIN_SIOC,
  .pin_d7 = CAM_PIN_D7,
  .pin_d6 = CAM_PIN_D6,
  .pin_d5 = CAM_PIN_D5,
  .pin_d4 = CAM_PIN_D4,
  .pin_d3 = CAM_PIN_D3,
  .pin_d2 = CAM_PIN_D2,
  .pin_d1 = CAM_PIN_D1,
  .pin_d0 = CAM_PIN_D0,
  .pin_vsync = CAM_PIN_VSYNC,
  .pin_href = CAM_PIN_HREF,
  .pin_pclk = CAM_PIN_PCLK,
  .xclk_freq_hz = 20000000,
  .ledc_timer = LEDC_TIMER_0,
  .ledc_channel = LEDC_CHANNEL_0,
  .pixel_format = PIXFORMAT_RGB888,
  .frame_size = FRAMESIZE_QVGA, // 320x240, будем ресайзить
  .jpeg_quality = 12,
  .fb_count = 1,
  .grab_mode = CAMERA_GRAB_LATEST
};

// TFLite Micro объекты
tflite::MicroErrorReporter micro_error_reporter;
tflite::AllOpsResolver resolver;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

void setup() {
  Serial.begin(115200);
  motors.setup();

  // Камера
  esp_err_t err = esp_camera_init(&camera_config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    while (1);
  }

  // Модель
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch!");
    while (1);
  }
  interpreter = new tflite::MicroInterpreter(model, resolver, tensor_arena, kTensorArenaSize, &micro_error_reporter);
  interpreter->AllocateTensors();
  input = interpreter->input(0);
  output = interpreter->output(0);
  Serial.println("[NN] Модель загружена и готова к инференсу!");
}

void loop() {
  // 1. Снять кадр с камеры
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed");
    delay(100);
    return;
  }

  // 2. Преобразовать изображение: ресайз до 160x120, float32, CHW, нормализация [0,1]
  // (Зависит от вашей камеры! Здесь пример для RGB888)
  // ВНИМАНИЕ: если ваша камера выдаёт другой формат, потребуется конвертация!
  // fb->width, fb->height, fb->buf (uint8_t*)

  // Пример: ресайз и нормализация (очень упрощённо, для реального кода используйте быстрый ресайз)
  for (int c = 0; c < IMG_C; ++c) {
    for (int y = 0; y < IMG_H; ++y) {
      for (int x = 0; x < IMG_W; ++x) {
        int src_x = x * fb->width / IMG_W;
        int src_y = y * fb->height / IMG_H;
        int src_idx = (src_y * fb->width + src_x) * 3 + c;
        input->data.f[c * IMG_H * IMG_W + y * IMG_W + x] = fb->buf[src_idx] / 255.0f;
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

  // 4. Управление моторами
  int steer = int(offset * 127.0f);
  int left  = BASE_SPEED - steer;
  int right = BASE_SPEED + steer;
  motors.runs(left/2.5, right/2.5);

  // 5. Отладка
  Serial.printf("[NN] offset=%.3f steer=%d L=%d R=%d\n", offset, steer, left, right);

  delay(20); // ~50 Гц
} 