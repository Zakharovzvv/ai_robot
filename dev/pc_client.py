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
import requests
from io import BytesIO

MODE = "run"            # ← "train"  или  "run"
ESP32_IP = '192.168.31.123'   # IP ESP32 из монитора
WS_PORT = 2222                # WebSocket порт (ранее RX_PORT)
CAM_PORT = 81                 # Порт для MJPEG стрима с ESP32-CAM
CAM_URL = f'http://{ESP32_IP}:{CAM_PORT}/stream' # MJPEG стрим с ESP32-CAM

# dataset writers
vw = None
log = None
if MODE == "train":
    os.makedirs('../dataset', exist_ok=True)
    ts = int(time.time())
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw  = cv2.VideoWriter(f'../dataset/run_{ts}.mp4', fourcc, 30, (160,120))
    log = csv.writer(open(f'../dataset/run_{ts}.csv','w',newline=''))

# нейросеть
if MODE == "run":
    ort_path = '../weights/line.ort'
    onnx_path = '../weights/line.onnx'
    if os.path.exists(ort_path):
        model_path = ort_path
        print("[INFO] Используется оптимизированная модель ORT:", ort_path)
    elif os.path.exists(onnx_path):
        model_path = onnx_path
        print("[INFO] Используется обычная ONNX-модель:", onnx_path)
    else:
        raise FileNotFoundError("Не найден ни ../weights/line.ort, ни ../weights/line.onnx")

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

class ESP32CamStream:
    def __init__(self, url):
        self.url = url
        self.bytes_buffer = bytes()
        self.session = requests.Session()
    
    def start(self):
        self.response = self.session.get(self.url, stream=True)
        if self.response.status_code != 200:
            raise Exception(f"Не удалось подключиться к камере ESP32-CAM: {self.response.status_code}")
        print("[INFO] Подключено к стриму ESP32-CAM")
        self.boundary = self.response.headers.get('Content-Type').split('boundary=')[1]
        self.boundary_bytes = bytes('--' + self.boundary, 'utf8')
        self.frame_start = b'\r\n\r\nContent-Type: image/jpeg\r\n\r\n'
    
    def read(self):
        try:
            for chunk in self.response.iter_content(chunk_size=1024):
                if not chunk:
                    continue
                
                self.bytes_buffer += chunk
                start_idx = self.bytes_buffer.find(self.frame_start)
                
                if start_idx != -1:
                    # Находим начало JPEG данных
                    jpeg_start = start_idx + len(self.frame_start)
                    
                    # Ищем конец текущего фрейма (начало следующего boundary)
                    next_boundary = self.bytes_buffer.find(self.boundary_bytes, jpeg_start)
                    
                    if next_boundary != -1:
                        # Извлекаем JPEG данные
                        jpeg_data = self.bytes_buffer[jpeg_start:next_boundary-2]  # -2 чтобы убрать \r\n перед boundary
                        
                        # Обновляем буфер, оставляя только данные после текущего фрейма
                        self.bytes_buffer = self.bytes_buffer[next_boundary:]
                        
                        # Декодируем JPEG в изображение
                        nparr = np.frombuffer(jpeg_data, np.uint8)
                        try:
                            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            if frame is not None:
                                return True, frame
                        except:
                            print("[WARN] Ошибка декодирования JPEG")
        
        except Exception as e:
            print(f"[ERROR] Ошибка чтения потока с камеры: {e}")
        
        return False, None
    
    def release(self):
        if hasattr(self, 'response'):
            self.response.close()
        self.session.close()

async def main_ws():
    uri = f"ws://{ESP32_IP}:{WS_PORT}"
    print(f"[INFO] Подключение к {uri} ...")
    async with websockets.connect(uri, max_size=16) as ws:
        print("[INFO] WebSocket соединение установлено!")
        
        # Подключаемся к MJPEG стриму ESP32-CAM
        cam = ESP32CamStream(CAM_URL)
        try:
            cam.start()
        except Exception as e:
            print(f"[ERROR] {e}")
            print(f"[INFO] Убедитесь, что прошивка ESP32-CAM настроена на MJPEG стриминг")
            return
        
        fps_t0 = time.time();  frames = 0;  packets=0; lost=0
        global vw, log
        while True:
            ok, frame = cam.read()
            if not ok:
                print("[WARN] Не удалось получить кадр с камеры")
                await asyncio.sleep(0.1)
                continue
                
            frame = cv2.resize(frame, (160, 120))
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
        
        cam.release()
        if MODE=="train" and vw is not None: vw.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main_ws())
