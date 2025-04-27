# === Параметры обучения (меняйте здесь) ===
batch_size = 256         # Размер батча
# (Чем больше batch_size, тем быстрее обучение и стабильнее градиенты, но выше требования к памяти. Обычно 64-512. Для небольших GPU/MPS — 64-256.)

learning_rate = 1e-3     # Скорость обучения
# (Главный параметр оптимизации. Слишком большой — модель не сойдётся, слишком маленький — обучение будет очень медленным. Обычно 1e-3 для Adam/AdamW, 1e-2 для SGD.)

max_epochs = 100         # Максимальное число эпох
# (Ограничение на число проходов по всему датасету. Если включён early stopping, обучение может завершиться раньше. Обычно 20-200.)

target_mae = 0.01        # Целевая ошибка MAE для остановки
# (Если средняя абсолютная ошибка (MAE) на обучении становится меньше этого значения — обучение останавливается. Подбирайте по задаче: чем меньше, тем точнее, но дольше.)

patience = 10            # Кол-во эпох без улучшения для early stopping (0 = не использовать)
# (Если MAE не улучшается patience эпох подряд — обучение останавливается. Помогает избежать переобучения и экономит время. Обычно 5-20. Если не нужен early stopping — ставьте 0.)

# =========================================

# обучение

#!/usr/bin/env python3
# test_4.py

import multiprocessing
from torch.multiprocessing import freeze_support
import time
import os

# Установка переменной окружения для решения проблемы с MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def main():
    import os
    import torch
    import torchvision as tv
    from torch.utils.data import DataLoader, TensorDataset
    import numpy as np
    import time
    global batch_size, learning_rate, max_epochs, target_mae, patience
    
    # Расширенная проверка доступных вычислительных устройств
    # 1. Сначала проверяем Apple Silicon (MPS)
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print(f"[INFO] Используется GPU (Apple Silicon) через MPS")
        print(f"[INFO] Включен режим PYTORCH_ENABLE_MPS_FALLBACK для обеспечения совместимости")
        use_jit = False  # Отключаем JIT для MPS из-за проблем совместимости
    # 2. Затем проверяем наличие NVIDIA GPU (CUDA)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Используется NVIDIA GPU через CUDA. GPU: {torch.cuda.get_device_name(0)}")
        use_jit = True  # Для CUDA JIT обычно работает хорошо
    # 3. Проверяем наличие AMD GPU (ROCm)
    elif hasattr(torch, 'hip') and torch.hip.is_available():
        device = torch.device("hip")
        print(f"[INFO] Используется AMD GPU через ROCm/HIP")
        use_jit = True
    # 4. Проверка для возможных других бэкендов GPU
    elif hasattr(torch, 'vulkan') and torch.vulkan.is_available():
        device = torch.device("vulkan")
        print(f"[INFO] Используется GPU через Vulkan backend")
        use_jit = False  # Для безопасности отключаем JIT для менее тестированных бэкендов
    # 5. В последнюю очередь используем CPU
    else:
        device = torch.device("cpu")
        print(f"[INFO] GPU не доступен, используется CPU")
        print("[СОВЕТ] Для максимальной производительности лучше запускать на компьютере с GPU")
        use_jit = True  # Для CPU JIT обычно работает нормально
    
    # Определяем количество доступных CPU ядер (для CPU или для загрузки данных)
    num_workers = multiprocessing.cpu_count()
    print(f"[INFO] Доступно CPU ядер: {num_workers}")

    # Настройка PyTorch для использования всех ядер CPU
    # (актуально для CPU или для загрузки данных)
    torch.set_num_threads(num_workers)
    print(f"[INFO] PyTorch будет использовать {torch.get_num_threads()} потоков для CPU-операций")

    # Удаляем попытку импорта onnxruntime_tools, так как этот пакет больше не доступен отдельно
    # В новых версиях onnxruntime функциональность оптимизации моделей включена в основной пакет
    HAVE_ORT = False
    try:
        import onnxruntime as ort
        HAVE_ORT = True
    except ImportError:
        pass

    # 0. ensure weights dir
    os.makedirs('weights', exist_ok=True)

    # 1. dataset
    print("[INFO] Загрузка датасета...")
    data = np.load('dataset/train.npz')
    X = torch.tensor(data['X'], dtype=torch.float32)
    y = torch.tensor(data['y'], dtype=torch.float32)
    ds = TensorDataset(X, y)
    print(f"[INFO] Датасет загружен: {len(ds)} примеров, размер входных данных: {X.shape}")

    # --- Определяем device_type для amp и pin_memory ---
    if device.type == 'cuda':
        amp_device = 'cuda'
        pin_memory = True
    elif device.type == 'mps':
        amp_device = 'mps'
        pin_memory = False
    else:
        amp_device = 'cpu'
        pin_memory = False

    # Увеличиваем batch_size и оптимизируем загрузку данных
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, 
                    num_workers=min(6, num_workers), 
                    pin_memory=pin_memory,  # Только для CUDA
                    persistent_workers=True,  # Держит рабочие процессы активными
                    prefetch_factor=2)  # Предзагрузка данных

    # 2. model
    model = tv.models.mobilenet_v2(width_mult=0.35)
    model.classifier[1] = torch.nn.Linear(1280, 1)
    
    # Перемещаем модель на GPU/CPU
    model = model.to(device)
    print(f"[INFO] Модель перемещена на {device}")

    # Можно попробовать использовать JIT-компиляцию модели для ускорения
    # но только если это подходит для выбранного устройства
    if use_jit:
        try:
            # Попытка JIT-компиляции модели
            dummy = torch.randn(1, 3, 120, 160, device=device)
            model = torch.jit.script(model)
            print("[INFO] Модель скомпилирована с помощью JIT")
        except Exception as e:
            print(f"[INFO] JIT-компиляция не удалась: {e}")
            # Продолжаем без JIT
    else:
        print("[INFO] JIT-компиляция отключена для текущего устройства")

    # --- Настройка mixed precision и GradScaler ---
    use_mixed_precision = device.type == 'cuda'
    try:
        import torch.amp
        have_amp = True
    except ImportError:
        have_amp = False

    if use_mixed_precision and have_amp:
        try:
            from torch.amp import GradScaler, autocast
            scaler = GradScaler()
            autocast_ctx = lambda: autocast(device_type='cuda')
            print(f"[INFO] Включена тренировка со смешанной точностью для ускорения (cuda)")
        except Exception:
            from torch.cuda.amp import GradScaler, autocast
            scaler = GradScaler()
            autocast_ctx = lambda: autocast()
            print(f"[INFO] Включена тренировка со смешанной точностью для ускорения (cuda, legacy)")
    else:
        scaler = None
        autocast_ctx = None
        if device.type == 'mps':
            print("[INFO] MPS: смешанная точность не поддерживается, обучение в обычном режиме")
        elif device.type == 'cpu':
            print("[INFO] CPU: смешанная точность не поддерживается, обучение в обычном режиме")

    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.L1Loss()

    # 3. train (быстро для проверки)
    total_start_time = time.time()  # Общее время обучения
    final_mae = 0.0  # Финальная ошибка
    best_mae = float('inf')
    patience_counter = 0
    mae_history = []  # Для хранения MAE по эпохам
    
    print("\n[ОБУЧЕНИЕ] Начало процесса обучения...")
    for epoch in range(1, max_epochs+1):
        model.train()
        start = time.time()  # начало эпохи
        total = 0.0
        total_batches = len(dl)
        print_interval = max(1, total_batches//5)  # Уменьшаем частоту вывода
        
        for i, (xb, yb) in enumerate(dl):
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            # Обучение со смешанной точностью
            if use_mixed_precision and autocast_ctx is not None:
                with autocast_ctx():
                    pred = model(xb).squeeze(1)
                    loss = loss_fn(pred, yb)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                pred = model(xb).squeeze(1)
                loss = loss_fn(pred, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            total += loss.item() * xb.size(0)
            if (i+1) % print_interval == 0:
                progress = (i+1) / total_batches * 100
                print(f"Эпоха {epoch}/{max_epochs} - {progress:.1f}% [батч {i+1}/{total_batches}]")
        elapsed = time.time() - start  # конец эпохи
        final_mae = total/len(ds)  # Сохраняем финальную ошибку
        mae_history.append(final_mae)  # Добавляем MAE в историю
        print(f"Эпоха {epoch}/{max_epochs} — MAE: {final_mae:.4f} — время: {elapsed:.1f} сек")
        # Early stopping по целевой ошибке
        if final_mae < target_mae:
            print(f"[EARLY STOP] Достигнута целевая ошибка MAE < {target_mae}")
            break
        # Early stopping по patience
        if patience > 0:
            if final_mae < best_mae - 1e-6:
                best_mae = final_mae
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[EARLY STOP] MAE не улучшается {patience} эпох подряд")
                    break

    # Выводим общую статистику обучения
    total_time = time.time() - total_start_time
    print(f"\n[РЕЗУЛЬТАТЫ] Обучение завершено!")
    print(f"[РЕЗУЛЬТАТЫ] Общее время обучения: {total_time:.1f} сек ({total_time/60:.1f} мин)")
    print(f"[РЕЗУЛЬТАТЫ] Итоговая ошибка MAE: {final_mae:.4f}")
    print(f"[РЕЗУЛЬТАТЫ] Среднее время на эпоху: {total_time/epoch:.1f} сек")

    # Построение графика MAE по эпохам
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,4))
        plt.plot(range(1, len(mae_history)+1), mae_history, marker='o')
        plt.xlabel('Эпоха')
        plt.ylabel('MAE')
        plt.title('MAE по эпохам обучения')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("[ВНИМАНИЕ] Для построения графика установите matplotlib: pip install matplotlib")

    # 4. save weights
    pth = 'weights/mnv2_035.pth'
    if hasattr(model, 'module'):  # Если модель была обернута в DataParallel
        torch.save(model.module.state_dict(), pth)
    else:
        torch.save(model.state_dict(), pth)
    print(f"[✔] Веса сохранены в {pth}")

  
if __name__ == "__main__":
    freeze_support()  # Для поддержки мультипроцессинга на Windows
    main()
