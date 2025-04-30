#!/usr/bin/env python3
"""
Экспорт обученной модели в несколько форматов:

- weights/line_scripted.pt  (TorchScript)
- weights/line.onnx         (ONNX)
- weights/line.ort          (оптимизированный ONNXRuntime)
- weights/line_int8.tflite  (TFLite int8 для микроконтроллеров)
- weights/model_data.cc     (C-массив для ESP32 / TFLite Micro)

Запуск:
    python export_models.py
"""

import os
import torch
import torchvision as tv
import subprocess

# ---------- 0. пути / опции ----------
os.makedirs('../weights', exist_ok=True)
DO_MICRO_C_ARRAY = True

# ---------- 1. загружаем обученные веса ----------
print("[i] Загружаем модель...")
model = tv.models.mobilenet_v2(width_mult=0.35)
model.classifier[1] = torch.nn.Linear(1280, 1)
model.load_state_dict(torch.load('../weights/mnv2_035.pth'))
model.eval()

# ---------- 2. TorchScript ----------
scripted = torch.jit.script(model.cpu())
scripted = torch.jit.optimize_for_inference(scripted)
scripted.save('../weights/line_scripted.pt')
print("[✔] TorchScript сохранён: ../weights/line_scripted.pt")

# ---------- 3. ONNX ----------
print("[i] Экспорт в ONNX...")
dummy = torch.randn(1, 3, 120, 160)
onnx_path = '../weights/line.onnx'
torch.onnx.export(
    model, dummy, onnx_path,
    input_names=['input'], output_names=['offset'],
    dynamic_axes={'input': {0: 'N'}, 'offset': {0: 'N'}},
    opset_version=12)
print("[✔] ONNX экспортирован:", onnx_path)

# ---------- 3.1 ORT-оптимизация (если установлен onnxruntime) ----------
try:
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    so.optimized_model_filepath = '../weights/line.ort'
    _ = ort.InferenceSession(onnx_path, so)
    print("[✔] ORT оптимизирован и сохранён: ../weights/line.ort")
except ImportError:
    print("[i] onnxruntime не установлен – пропускаем ORT-оптимизацию")

# ---------- 4. ONNX → TensorFlow → TFLite int8 ----------
# print("[i] Конвертация ONNX→TensorFlow→TFLite int8...")

# try:
#     import onnx
#     from onnx_tf.backend import prepare
#     import tensorflow as tf

#     # 4.1 ONNX → TensorFlow SavedModel
#     tf_path = '../weights/line_tf'
#     onnx_model = onnx.load(onnx_path)
#     tf_rep = prepare(onnx_model)
#     tf_rep.export_graph(tf_path)
#     print("[✔] TensorFlow SavedModel экспортирован:", tf_path)

#     # 4.2 TensorFlow SavedModel → TFLite int8
#     converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
#     converter.optimizations = [tf.lite.Optimize.DEFAULT]  # посттренировочная квантизация
#     tflite_int8 = converter.convert()
#     tflite_path = '../weights/line_int8.tflite'
#     with open(tflite_path, 'wb') as f:
#         f.write(tflite_int8)
#     print("[✔] TFLite int8 модель сохранена:", tflite_path)

# except ImportError as e:
#     print("[!] Ошибка импорта для ONNX-TF или TensorFlow:", e)
#     print("[!] Пропускаем конвертацию в TFLite.")

# ---------- 5. (опц.) TFLite-Micro C-array ----------
if DO_MICRO_C_ARRAY and os.path.exists('../weights/line_int8.tflite'):
    print("[i] Генерация C-массива для TFLite Micro...")
    cc_path = '../weights/model_data.cc'
    with open(cc_path, 'w') as fcc, open('../weights/line_int8.tflite', 'rb') as ftflite:
        import textwrap, binascii
        data = binascii.hexlify(ftflite.read()).decode()
        arr = ','.join(f'0x' + data[i:i+2] for i in range(0, len(data), 2))
        fcc.write('#include <cstdint>\nconst unsigned char g_model[] = {\n')
        fcc.write(textwrap.fill(arr, width=90))
        fcc.write('\n};\nconst unsigned int g_model_len = sizeof(g_model);')
    print("[✔] Массив для TFLite Micro сохранён:", cc_path)

print("\n[ALL DONE ✅]")
