# подготовка датасета (постобработка видео для дальнейшего обучения)

import cv2, csv, glob, numpy as np, os
X, y = [], []
for mp4 in glob.glob('dataset/*.mp4'):
    csv_path = mp4.replace('.mp4', '.csv')
    err = {float(t): float(e) for t, e in csv.reader(open(csv_path))}
    cap = cv2.VideoCapture(mp4)
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
np.savez_compressed('dataset/train.npz', X=np.array(X), y=np.array(y))