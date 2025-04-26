#!/usr/bin/env python3
"""
pc_client.py – логирование и/или управление роботом

MODE = "train"  – записываем видео + ошибки
MODE = "run"    – запускаем нейросеть и шлём steer
"""

MODE = "run"            # ← "train"  или  "run"
ESP32_IP = ('192.168.31.123', 2222)   # IP ESP32 из монитора
CAM_URL  = 'http://192.168.31.137:4747/video'
UDP_PORT = 3333                         # error от ESP32

import cv2, numpy as np, socket, struct, time, os, csv
import onnxruntime as ort

# UDP
sock_rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_rx.bind(('', UDP_PORT))
sock_rx.setblocking(False)
sock_tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Камера
cap = cv2.VideoCapture(CAM_URL)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# dataset writers
if MODE == "train":
    os.makedirs('dataset', exist_ok=True)
    ts = int(time.time())
    vw  = cv2.VideoWriter(f'dataset/run_{ts}.mp4', fourcc, 30, (160,120))
    log = csv.writer(open(f'dataset/run_{ts}.csv','w',newline=''))

# нейросеть
if MODE == "run":
    ort_path = 'weights/line.ort'
    onnx_path = 'weights/line.onnx'
    if os.path.exists(ort_path):
        model_path = ort_path
        print("[INFO] Используется оптимизированная модель ORT:", ort_path)
    elif os.path.exists(onnx_path):
        model_path = onnx_path
        print("[INFO] Используется обычная ONNX-модель:", onnx_path)
    else:
        raise FileNotFoundError("Не найден ни weights/line.ort, ни weights/line.onnx")

    available_providers = ort.get_available_providers()
    print("[INFO] Доступные ONNXRuntime провайдеры:", available_providers)
    preferred = [
        "CoreMLExecutionProvider",  # для Mac
        "CUDAExecutionProvider",    # для NVIDIA GPU
        "CPUExecutionProvider"
    ]
    providers = [p for p in preferred if p in available_providers]
    if not providers:
        providers = available_providers  # fallback

    ort_sess = ort.InferenceSession(model_path, providers=providers)
    offset_prev = 0.0

fps_t0 = time.time();  frames = 0;  packets=0; lost=0
print("[START]", MODE.upper())
while True:
    ok, frame = cap.read()
    if not ok: break
    frame = cv2.resize(frame,(160,120))

    # --- RX error ---
    try:
        data,_ = sock_rx.recvfrom(1)
        err = struct.unpack('b', data)[0]/127.0
        packets += 1
    except BlockingIOError:
        err = None
        lost += 1

    if MODE=="train" and err is not None:
        vw.write(frame);  log.writerow([time.time(), err])

    # --- RUN: инференс и steer ---
    if MODE=="run":
        inp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype('float32')/255
        inp = np.transpose(inp,(2,0,1))[None]
        offset = ort_sess.run(None,{'input':inp})[0][0][0]
        offset = 0.6*offset_prev + 0.4*offset;  offset_prev = offset
        steer = int(np.clip(offset*127,-127,127))
        sock_tx.sendto(struct.pack('b',steer), ESP32_IP)
        cv2.putText(frame,f"steer={steer}",(5,15),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
    else:
        if err is not None:
            cv2.putText(frame,f"err={err:+.2f}",(5,15),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

    cv2.imshow(MODE, frame)
    if cv2.waitKey(1)&0xFF==27: break

    # --- FPS / stats each 2 s ---
    frames += 1
    if time.time()-fps_t0 > 2:
        fps = frames / (time.time()-fps_t0)
        print(f"[STAT] fps={fps:.1f}  pkts={packets}  lost={lost}")
        fps_t0=time.time(); frames=packets=lost=0

cap.release(); sock_rx.close(); sock_tx.close()
if MODE=="train": vw.release()
cv2.destroyAllWindows()
