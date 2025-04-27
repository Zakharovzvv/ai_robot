| № | Класс / Плата                  | CPU / NPU                         | ОЗУ / Flash       | TOPS*   | FPS (160×120, MobileNet-0.35) | Типичное питание | Поддерживаемые фреймворки | Что «потянет»        | Стоимость (USD) | Рекомендации |
|---|----------------------------------|------------------------------------|-------------------|---------|----------------------------------|------------------|---------------------------|------------------|----------------|--------------|
| 1 | STM32 F407 (или Blue-Pill)       | 168 МГц Cortex-M4, FPU          | 192 KB / 1 MB     | —     | ~2 FPS                           | 0.15 W           | TFLite Micro (int8)        | 1-2 сл CNN        | 3–10           | Минимализм    |
| 2 | RP2040 (Raspberry Pi Pico)       | 2 × 133 МГц Cortex-M0+         | 264 KB / ext Flash | —     | 4-6 FPS                          | 0.2 W            | TFLite Micro              | tiny-CNN          | 5–10           | Датчики+SPI камера |
| 3 | ESP32-S3                         | 2 × 240 МГц Xtensa LX7, SIMD    | 512 KB + 8 MB PSRAM| 0.32    | 18-25 FPS                        | 0.6-1 W          | TFLite Micro (+Esp-DL)     | MobileNet-0.25    | 8–12           | Камера+AI оптимальный |
| 4 | Kendryte K210 (Maix Dock)         | 400 МГц RISC-V + KPU              | 8 MB SRAM         | 1 TOPS  | 50-60 FPS                        | 0.9 W            | nncase, Micropython        | MobileNet-0.25    | 25–35          | Устаревающий   |
| 5 | Bouffalo BL808/BL70X              | 480 МГц RISC-V + DLA              | 24 MB PSRAM       | 3 TOPS  | 70 FPS                           | 1 W              | TFLite Micro (SDK)         | MobileNet-0.5     | 12–20           | Новый K210     |
| 6 | Raspberry Pi Zero 2 W             | 4 × 1 ГГц Cortex-A53           | 512 MB LPDDR2     | —     | 35 FPS                           | 1.8 W            | TFLite Lite, PyTorch Lite  | MobileNet-V2-0.35 | 20–25           | Бюджетный Linux |
| 7 | OpenMV Cam H7 Plus                | 480 МГц Cortex-M7 + 2 MB SRAM      | SDRAM 32 MB       | —     | 40 FPS                           | 1 W              | MicroPython (OpenMV)       | Line-tracking     | 70–90           | Все-в-одном IDE |
| 8 | Raspberry Pi 5 + AI Kit (Hailo-8L) | 4 × 2.4 ГГц A76 + NPU 13 TOPS   | 4-8 GB LPDDR4X    | 13 TOPS | 250 FPS                         | 5-7 W            | ONNX RT, PyTorch, TFLite   | YOLOv8-n          | 150–200         | Универсальный   |
| 9 | Jetson Nano 2 GB                  | 4 × 1.4 ГГц Cortex-A57 + 128-CUDA| 2 GB              | 0.5 TOPS| 90 FPS                           | 5 W              | TensorRT, PyTorch          | MobileNet-V2-1.0  | 60–80           | Стареющий SBC  |
|10 | Jetson Orin Nano 8 GB              | 6 × A78AE + 1024 CUDA + 32 Tensor Cores    | 8 GB              | 40 TOPS | 600 FPS                         | 10-15 W          | TensorRT, ONNX RT          | YOLOv8-m          | 400–500         | Мобильный AI   |
|11 | Jetson Orin NX 16 GB               | 8 × A78AE + 2048 CUDA                     | 16 GB             | 100 TOPS| >1 kFPS                         | 15-25 W          | ROS 2 + Isaac, TRT          | SLAM + CV         | 600–800         | RoboCup, R&D    |
|12 | Jetson AGX Orin 64 GB              | 12 × A78AE + 2048 CUDA                     | 64 GB             | 275 TOPS| >2 kFPS                         | 30-60 W          | Full-stack AI, Robotics    | Многосенсорная автономия | 2000–2500    | Промышленный R&D |


(*) TOPS — пиковая производительность в INT8.

