# подготовка датасета (постобработка видео для дальнейшего обучения)

import cv2, csv, glob, numpy as np, os
X, y = [], []
mp4_files = glob.glob('../dataset/*.mp4')
print(f"[INFO] Найдено {len(mp4_files)} видео для обработки.")
for idx, mp4 in enumerate(mp4_files, 1):
    print(f"[INFO] ({idx}/{len(mp4_files)}) Обработка видео: {os.path.basename(mp4)}")
    csv_path = mp4.replace('.mp4', '.csv')
    if not os.path.exists(csv_path):
        print(f"[WARN] CSV-файл не найден для {mp4}, пропуск.")
        continue
    print(f"[INFO] Чтение ошибок из: {os.path.basename(csv_path)}")
    err = {float(t): float(e) for t, e in csv.reader(open(csv_path))}
    cap = cv2.VideoCapture(mp4)
    frame_count = 0
    while True:
        ok, frame = cap.read(); ts = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
        if not ok: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype('float32')
        # Централизованная нормализация (-1 до +1) вместо [0, 1]
        frame = (frame / 255.0 - 0.5) / 0.5
        frame = np.transpose(frame, (2,0,1))  # CHW
        # ищем ближайшую ошибку по времени
        key = min(err, key=lambda k: abs(k-ts))
        X.append(frame); y.append(err[key])
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"    [INFO] Обработано кадров: {frame_count}")
    print(f"[INFO] Всего обработано кадров из {os.path.basename(mp4)}: {frame_count}")
print(f"[INFO] Всего кадров в датасете: {len(X)}")
out_path = '../dataset/train.npz'
np.savez_compressed(out_path, X=np.array(X), y=np.array(y))
print(f"[INFO] Датасет сохранён: {out_path}")