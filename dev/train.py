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

val_ratio = 0.2          # Доля данных для валидации (0.0-0.5)
# (Часть датасета, которая будет использоваться для проверки качества модели. Обычно 0.1-0.2 (10-20%). 0.0 = не использовать валидацию.)

weight_decay = 1e-4      # Коэффициент L2-регуляризации
# (Помогает предотвратить переобучение, "штрафуя" большие веса. Обычно 1e-4 - 1e-5. 0 = отключить регуляризацию.)

# =========================================

#!/usr/bin/env python3
import os
import torch
import torchvision as tv
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import time
import multiprocessing

# Опциональные импорты
try:
    import matplotlib.pyplot as plt
    HAVE_MATPLOTLIB = True
except ImportError:
    HAVE_MATPLOTLIB = False

try:
    import onnxruntime as ort
    HAVE_ORT = True
except ImportError:
    HAVE_ORT = False

from torch.amp import GradScaler, autocast

# Установка переменной окружения для решения проблемы с MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def select_device():
    """Выбор оптимального вычислительного устройства"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"[INFO] Используется GPU (Apple Silicon) через MPS")
        print(f"[INFO] Включен режим PYTORCH_ENABLE_MPS_FALLBACK для совместимости")
        return device, False  # Второй параметр - использование JIT (для MPS отключено)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Используется NVIDIA GPU через CUDA. GPU: {torch.cuda.get_device_name(0)}")
        return device, True
    else:
        device = torch.device("cpu")
        print(f"[INFO] GPU не доступен, используется CPU")
        print("[СОВЕТ] Для максимальной производительности лучше запускать на компьютере с GPU")
        return device, True

def main():
    global batch_size, learning_rate, max_epochs, target_mae, patience, val_ratio, weight_decay
    
    # Выбор вычислительного устройства
    device, use_jit = select_device()
    
    # Определяем количество доступных CPU ядер для загрузки данных
    num_workers = min(6, multiprocessing.cpu_count())
    print(f"[INFO] Доступно CPU ядер: {multiprocessing.cpu_count()}, будет использовано: {num_workers}")

    # Настройка PyTorch для использования всех ядер CPU
    torch.set_num_threads(num_workers)
    print(f"[INFO] PyTorch будет использовать {torch.get_num_threads()} потоков для CPU-операций")

    # 0. Создаем необходимые директории для сохранения моделей и графиков
    os.makedirs('../weights', exist_ok=True)
    print("[INFO] Проверена директория ../weights")

    # 1. dataset
    print("[INFO] Загрузка датасета...")
    data = np.load('../dataset/train.npz')
    X = torch.tensor(data['X'], dtype=torch.float32)
    y = torch.tensor(data['y'], dtype=torch.float32)
    ds = TensorDataset(X, y)
    print(f"[INFO] Датасет загружен: {len(ds)} примеров, размер входных данных: {X.shape}")
    
    # Разделение на обучающую и валидационную выборки
    if val_ratio > 0:
        val_size = int(len(ds) * val_ratio)
        train_size = len(ds) - val_size
        train_ds, val_ds = random_split(ds, [train_size, val_size], 
                                        generator=torch.Generator().manual_seed(42))
        print(f"[INFO] Датасет разделен: {train_size} примеров для обучения, {val_size} для валидации")
    else:
        train_ds = ds
        val_ds = None
        print("[INFO] Валидация отключена (val_ratio=0)")

    # --- Определяем параметры для DataLoader ---
    pin_memory = device.type == 'cuda'  # Только для CUDA эффективно
    persistent_workers = num_workers > 0
    prefetch_factor = 2 if num_workers > 0 else None

    # Увеличиваем batch_size и оптимизируем загрузку данных
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                    num_workers=num_workers, 
                    pin_memory=pin_memory,
                    persistent_workers=persistent_workers,
                    prefetch_factor=prefetch_factor)
    
    # Создаем валидационный DataLoader, если нужно
    if val_ds is not None:
        val_dl = DataLoader(val_ds, batch_size=batch_size*2, shuffle=False,
                     num_workers=num_workers,
                     pin_memory=pin_memory)

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

    # --- Настройка mixed precision ---
    use_mixed_precision = device.type == 'cuda'
    scaler = None
    autocast_ctx = None
    
    if use_mixed_precision:
        try:
            scaler = GradScaler()
            autocast_ctx = lambda: autocast(device_type='cuda')
            print(f"[INFO] Включена тренировка со смешанной точностью для ускорения (cuda)")
        except Exception as e:
            print(f"[INFO] Ошибка инициализации смешанной точности: {e}")
    else:
        print(f"[INFO] {device.type.upper()}: смешанная точность не поддерживается, обучение в обычном режиме")

    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = torch.nn.L1Loss()

    # 3. train (быстро для проверки)
    total_start_time = time.time()  # Общее время обучения
    final_train_mae = 0.0  # Финальная ошибка на обучении
    final_val_mae = 0.0    # Финальная ошибка на валидации
    best_val_mae = float('inf')
    patience_counter = 0
    train_mae_history = []  # Для хранения MAE по эпохам (обучение)
    val_mae_history = []    # Для хранения MAE по эпохам (валидация)
    prev_train_mae = float('inf')
    prev_val_mae = float('inf')
    
    print("\n[ОБУЧЕНИЕ] Начало процесса обучения...")
    for epoch in range(1, max_epochs+1):
        # ----- ОБУЧЕНИЕ -----
        model.train()
        start = time.time()  # начало эпохи
        total_train_loss = 0.0
        total_train_batches = len(train_dl)
        print_interval = max(1, total_train_batches//5)  # Уменьшаем частоту вывода
        
        for i, (xb, yb) in enumerate(train_dl):
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
            total_train_loss += loss.item() * xb.size(0)
            if (i+1) % print_interval == 0:
                progress = (i+1) / total_train_batches * 100
                print(f"Эпоха {epoch}/{max_epochs} - {progress:.1f}% [батч {i+1}/{total_train_batches}]")
        
        # ----- ВАЛИДАЦИЯ -----
        model.eval()
        total_val_loss = 0.0
        if val_ds is not None:
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                    pred = model(xb).squeeze(1)
                    loss = loss_fn(pred, yb)
                    total_val_loss += loss.item() * xb.size(0)
        
        elapsed = time.time() - start  # конец эпохи
        final_train_mae = total_train_loss/len(train_ds)  # Сохраняем финальную ошибку обучения
        train_mae_history.append(final_train_mae)  # Добавляем MAE в историю обучения
        
        # Считаем метрики валидации, если есть валидационный набор
        status_line = f"Эпоха {epoch}/{max_epochs} — Train MAE: {final_train_mae:.4f}"
        if val_ds is not None:
            final_val_mae = total_val_loss/len(val_ds)
            val_mae_history.append(final_val_mae)
            status_line += f" — Val MAE: {final_val_mae:.4f}"
            
            # Отслеживание переобучения (Train MAE падает, Val MAE растёт)
            if epoch > 1 and final_train_mae < prev_train_mae and final_val_mae > prev_val_mae:
                status_line += " ⚠️ ПЕРЕОБУЧЕНИЕ"
                
        status_line += f" — время: {elapsed:.1f} сек"
        print(status_line)
        
        # Сохраняем текущие значения для отслеживания тренда
        prev_train_mae = final_train_mae
        prev_val_mae = final_val_mae
        
            
        # Early stopping по patience (на валидации, если включена)
        if patience > 0 and val_ds is not None:
            if final_val_mae < best_val_mae - 1e-6:
                best_val_mae = final_val_mae
                patience_counter = 0
                
                # Сохраняем лучшую модель на валидации
                pth = '../weights/mnv2_035_best.pth'
                if hasattr(model, 'module'):
                    torch.save(model.module.state_dict(), pth)
                else:
                    torch.save(model.state_dict(), pth)
                print(f"[✓] Сохранена лучшая модель на валидации: {pth}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[EARLY STOP] Val MAE не улучшается {patience} эпох подряд")
                    break
        # Early stopping по patience (на обучении, если валидация выключена)
        elif patience > 0:
            if final_train_mae < best_val_mae - 1e-6:
                best_val_mae = final_train_mae
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[EARLY STOP] Train MAE не улучшается {patience} эпох подряд")
                    break

    # Выводим общую статистику обучения
    total_time = time.time() - total_start_time
    print(f"\n[РЕЗУЛЬТАТЫ] Обучение завершено!")
    print(f"[РЕЗУЛЬТАТЫ] Общее время обучения: {total_time:.1f} сек ({total_time/60:.1f} мин)")
    print(f"[РЕЗУЛЬТАТЫ] Итоговая ошибка Train MAE: {final_train_mae:.4f}")
    if val_ds is not None:
        print(f"[РЕЗУЛЬТАТЫ] Итоговая ошибка Val MAE: {final_val_mae:.4f}")
    print(f"[РЕЗУЛЬТАТЫ] Среднее время на эпоху: {total_time/epoch:.1f} сек")

    # Построение графика MAE по эпохам
    try:
        plt.figure(figsize=(8,4))
        plt.plot(range(1, len(train_mae_history)+1), train_mae_history, marker='o', label='Train MAE')
        if val_ds is not None:
            plt.plot(range(1, len(val_mae_history)+1), val_mae_history, marker='s', label='Validation MAE')
            plt.legend()
        plt.xlabel('Эпоха')
        plt.ylabel('MAE')
        plt.title('MAE по эпохам обучения')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('../weights/training_curve.png')  # Сохраняем график
        print(f"[✓] График обучения сохранен: ../weights/training_curve.png")
        plt.show()
    except ImportError:
        print("[ВНИМАНИЕ] Для построения графика установите matplotlib: pip install matplotlib")
    except Exception as e:
        print(f"[ОШИБКА] Не удалось построить график: {e}")

    # 4. save weights (последняя модель)
    pth = '../weights/mnv2_035.pth'
    if hasattr(model, 'module'):  # Если модель была обернута в DataParallel
        torch.save(model.module.state_dict(), pth)
    else:
        torch.save(model.state_dict(), pth)
    print(f"[✔] Финальные веса сохранены в {pth}")

    # Загружаем лучшую модель по валидации, если она была сохранена
    if patience > 0 and val_ds is not None and os.path.exists('../weights/mnv2_035_best.pth'):
        print("[INFO] Загружена лучшая модель по валидации для экспорта")
        if hasattr(model, 'module'):
            model.module.load_state_dict(torch.load('../weights/mnv2_035_best.pth'))
        else:
            model.load_state_dict(torch.load('../weights/mnv2_035_best.pth'))

if __name__ == "__main__":
    main()
