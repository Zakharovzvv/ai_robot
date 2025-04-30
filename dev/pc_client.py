#!/usr/bin/env python3
"""
pc_client.py – логирование и/или управление роботом через WebSocket

MODE = "train"  – записываем видео + ошибки
MODE = "run"    – запускаем нейросеть и шлём steer
"""

import asyncio
import websockets
import cv2, numpy as np, struct, time, os, csv
import onnxruntime as ort

MODE = "run"            # ← "train"  или  "run"
ESP32_IP = '192.168.31.123'   # IP ESP32 из монитора
WS_PORT = 2222                # WebSocket порт (ранее RX_PORT)
CAM_URL  = 'http://192.168.31.137:4747/video'

# dataset writers
vw = None
log = None
if MODE == "train":
    os.makedirs('dataset', exist_ok=True)
    ts = int(time.time())
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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

async def main_ws():
    uri = f"ws://{ESP32_IP}:{WS_PORT}"
    print(f"[INFO] Подключение к {uri} ...")
    async with websockets.connect(uri, max_size=16) as ws:
        print("[INFO] WebSocket соединение установлено!")
        cap = cv2.VideoCapture(CAM_URL)
        fps_t0 = time.time();  frames = 0;  packets=0; lost=0
        global vw, log
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.resize(frame,(160,120))
            err = None
            # --- RX error (train) ---
            if MODE=="train":
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.01)
                    if isinstance(msg, bytes) and len(msg)==1:
                        err = struct.unpack('b', msg)[0]/127.0
                        packets += 1
                    else:
                        err = None
                        lost += 1
                except asyncio.TimeoutError:
                    err = None
                    lost += 1
                if err is not None:
                    vw.write(frame);  log.writerow([time.time(), err])
                    cv2.putText(frame,f"err={err:+.2f}",(5,15),
                                cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
            # --- RUN: инференс и steer ---
            if MODE=="run":
                inp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype('float32')
                # Централизованная нормализация (-1 до +1) вместо [0, 1]
                inp = (inp / 255.0 - 0.5) / 0.5
                inp = np.transpose(inp,(2,0,1))[None]
                offset = ort_sess.run(None,{'input':inp})[0][0][0]
                offset = 0.6*offset_prev + 0.4*offset;  offset_prev = offset
                steer = int(np.clip(offset*127,-127,127))
                await ws.send(struct.pack('b',steer))
                cv2.putText(frame,f"steer={steer}",(5,15),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
                # (опционально) принимать ошибку от робота:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.01)
                    if isinstance(msg, bytes) and len(msg)==1:
                        err = struct.unpack('b', msg)[0]/127.0
                        packets += 1
                        cv2.putText(frame,f"err={err:+.2f}",(5,35),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
                except asyncio.TimeoutError:
                    pass
            cv2.imshow(MODE, frame)
            if cv2.waitKey(1)&0xFF==27: break
            # --- FPS / stats each 2 s ---
            frames += 1
            if time.time()-fps_t0 > 2:
                fps = frames / (time.time()-fps_t0)
                print(f"[STAT] fps={fps:.1f}  pkts={packets}  lost={lost}")
                fps_t0=time.time(); frames=packets=lost=0
        cap.release()
        if MODE=="train" and vw is not None: vw.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main_ws())
