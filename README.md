# AI Robot Project

Проект управления роботом с использованием компьютерного зрения и нейронной сети.

## Архитектура проекта

```
+-------------------+         Wi-Fi/WebSocket         +-------------------+
|                   | <----------------------------> |                   |
|   ПК (pc_client)  |                                |   Робот (ESP32)    |
|-------------------|                                |-------------------|
|  Камера (IP/USB)  |                                |  Моторы, сенсоры   |
|  Нейросеть (ORT)  |                                |  Линейка, ИК       |
|  Логирование      |                                |  WebSocket-сервер  |
|  WebSocket-клиент |                                |                   |
+-------------------+                                +-------------------+
        |                                                    |
        |                                                    |
        v                                                    v
  +-------------------+                              +-------------------+
  |   dataset/        |                              |  Управление,      |
  |   weights/        |                              |  передача ошибок  |
  +-------------------+                              +-------------------+
```

- **ПК (`dev/pc_client.py`)**: захват видео с камеры, инференс нейросети, отправка управляющих команд (steer) на ESP32, приём ошибок, логирование данных для обучения.
- **Робот (ESP32)**: отдельные прошивки для разных режимов:
  - **data_collection_robot.cpp** — сбор данных: едет по линии, отправляет ошибку на ПК.
  - **remote_control_robot.cpp** — дистанционное управление: принимает steer-команды с ПК по WebSocket.
  - **autonomous_robot.cpp** — автономный режим: использует встроенную нейросеть (TFLite Micro), управляет моторами без ПК.
- **Взаимодействие**: обмен данными между ПК и ESP32 происходит по WebSocket (порт 2222). Видео поток идёт отдельно (IP-камера или USB).
- **Данные**: все логи и видео сохраняются в `dataset/`, веса моделей — в `weights/`.

## Структура проекта

- `dev/pc_client.py` — клиент для логирования (режим train) и управления роботом (режим run) через нейросеть
- `dev/train.py` — обучение нейронной сети на собранном датасете
- `dev/export_models.py` — экспорт и оптимизация обученной модели (TorchScript, ONNX, ORT, TFLite)
- `dev/prepare_dateset.py` — подготовка датасета из видео и логов
- `dev/robots/esp32/data_collection_robot.cpp` — прошивка для сбора данных (TRAINING_MODE)
- `dev/robots/esp32/remote_control_robot.cpp` — прошивка для дистанционного управления с ПК (INFERENCE_MODE)
- `dev/robots/esp32/autonomous_robot.cpp` — автономная прошивка с нейросетью (TFLite Micro, без ПК)
- `dataset/` — директория с данными для обучения и тестирования (видео, логи, npz)
- `weights/` — директория с весами и экспортированными моделями (pth, onnx, ort, pt, tflite, cc)
- `requirements.txt` — список зависимостей Python

## Описание файлов для ESP32

- **data_collection_robot.cpp** — автономно едет по линии, отправляет ошибку на ПК для сбора датасета.
- **remote_control_robot.cpp** — принимает steer-команды по WebSocket от ПК, управляет моторами, используется для тестирования и инференса с ПК.
- **autonomous_robot.cpp** — полностью автономный режим: ESP32 снимает изображение с камеры, подаёт в встроенную нейросеть (TFLite Micro), управляет моторами без участия ПК.

## Описание папок dataset/ и weights/

### 📁 dataset/

**Назначение:**
Папка `dataset/` содержит все данные, необходимые для обучения и тестирования нейронной сети. Это "сырьё" для машинного обучения.

**Типичные файлы:**
- `runX.mp4` — видеозаписи с камеры робота, снятые во время сбора данных (например, `run1.mp4`). Используются для визуального анализа и для последующей разметки.
- `runX.csv` — CSV-файлы с логами ошибок управления, записанные синхронно с видео (например, `run1.csv`). Содержат временные метки и значения ошибок/отклонений, которые использовались для обучения.
- `train.npz` — готовый датасет в формате NumPy (сжатый архив), полученный после обработки видео и логов. Обычно содержит массивы признаков (X) и целевых значений (y), которые подаются на вход нейросети при обучении.

**Пример структуры:**
```
dataset/
├── run1.mp4         # Видео с камеры
├── run1.csv         # Логи ошибок для этого видео
├── train.npz        # Готовый датасет для обучения
```

**Для чего используется:**
- Сбор и хранение "сырых" данных с робота.
- Подготовка обучающего датасета для нейросети.
- Возможность повторного обучения или тестирования на одних и тех же данных.

---

### 📁 weights/

**Назначение:**
Папка `weights/` содержит все файлы с весами и экспортированными моделями, полученными после обучения. Это "результаты" машинного обучения.

**Типичные файлы:**
- `mnv2_035.pth` — веса обученной модели PyTorch (формат `.pth`). Используются для загрузки и дообучения модели в PyTorch.
- `line_scripted.pt` — TorchScript-версия модели (формат `.pt`). Удобна для быстрого инференса в PyTorch без исходного кода.
- `line.onnx` — модель в формате ONNX — универсальный формат для переноса между разными фреймворками и устройствами.
- `line.ort` — оптимизированная версия ONNX-модели для ONNX Runtime (быстрый инференс на CPU/GPU/ARM).
- `line_int8.tflite` — модель в формате TFLite с квантизацией int8 — для запуска на микроконтроллерах и мобильных устройствах.
- `model_data.cc` — C-массив с моделью для интеграции в прошивку микроконтроллера (например, ESP32 с TFLite Micro).

**Пример структуры:**
```
weights/
├── mnv2_035.pth         # Веса PyTorch
├── line_scripted.pt     # TorchScript-модель
├── line.onnx            # ONNX-модель
├── line.ort             # Оптимизированная ONNX Runtime модель
├── line_int8.tflite     # TFLite int8-модель
├── model_data.cc        # C-массив для микроконтроллера
```

**Для чего используется:**
- Хранение результатов обучения и экспорта моделей.
- Быстрый запуск инференса на разных устройствах (ПК, SBC, микроконтроллеры).
- Переносимость между платформами и фреймворками.

---

## Настройка окружения

1. Убедитесь, что у вас установлен Python 3.8+ и virtualenv
2. Выполните скрипт установки:

```bash
./scripts/setup.sh
```

Или выполните следующие шаги вручную:

```bash
# Активация виртуальной среды
source venv/bin/activate
# или
source venv310/bin/activate

# Установка зависимостей
pip install -r requirements.txt
```

## Запуск проекта

1. Активируйте виртуальную среду:

```bash
source venv/bin/activate
```

2. Запустите нужный скрипт, например (из каталога dev/):

```bash
# Для сбора данных и/или управления роботом (режимы train/run)
python pc_client.py

# Для подготовки датасета из видео и логов
python prepare_dateset.py

# Для обучения модели
python train.py

# Для экспорта и оптимизации модели
python export_models.py
```

## Примечания по настройке

- В файле `dev/pc_client.py` настройте `CAM_URL`, `ESP32_IP` под вашу сеть и робота
- В файле `dev/train.py` можно изменить параметры обучения и путь к датасету
- В файле `dev/export_models.py` настраивается экспорт моделей и форматы вывода
- В директории `dataset/` хранятся видео, логи и npz-файлы для обучения
- В директории `weights/` сохраняются веса и экспортированные модели
- Для работы с ESP32 используйте код из `dev/robots/esp32/` 