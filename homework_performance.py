import time
import torch

print("=" * 65)
print("ЗАДАНИЕ 3: Сравнение скорости CPU и GPU")
print("=" * 65)

# 1. Выбираем устройство: видеокарта или процессор
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")  # Видеокарта на Mac
else:
    device = torch.device("cpu")  # Процессор

print(f"Считаем на: {device}")




# 2. Создаем две большие таблицы со случайными числами
shape = (128, 512, 512)
A_cpu = torch.randn(shape)
B_cpu = torch.randn(shape)

# Отправляем копии таблиц на видеокарту
A_gpu = A_cpu.to(device)
B_gpu = B_cpu.to(device)





# 3. Секундомер: замеряет время работы операции в миллисекундах
def get_time(operation, A, B, is_gpu=False):
    # Разминка (warm-up): один холостой запуск для прогрева
    operation(A, B)
    if is_gpu and device.type == "mps":
        torch.mps.synchronize()

    # Засекаем время
    start = time.perf_counter()
    for _ in range(5):  # 5 раз для точности
        operation(A, B)
        if is_gpu and device.type == "mps":
            torch.mps.synchronize()
    end = time.perf_counter()

    # Среднее время в миллисекундах (мс)
    return ((end - start) / 5) * 1000





# 4. Список из 5 операций из задания
def op_matmul(a, b): return torch.matmul(a, b)  # Матричное умножение
def op_add(a, b):    return a + b               # Поэлементное сложение
def op_mul(a, b):    return a * b               # Поэлементное умножение
def op_trans(a, b):  return a.transpose(-1, -2) # Переворот матрицы
def op_sum(a, b):    return a.sum()             # Сумма всех чисел

tasks = [
    ("Матричное умножение (@)", op_matmul),
    ("Поэлементное сложение (+)", op_add),
    ("Поэлементное умножение (*)", op_mul),
    ("Транспонирование", op_trans),
    ("Сумма всех элементов", op_sum),
]

# 5. Запуск гонки и вывод таблицы
print("-" * 65)
print(f"{'Операция':<28} | {'CPU (мс)':<10} | {'GPU (мс)':<10} | Ускорение")
print("-" * 65)

for name, func in tasks:
    t_cpu = get_time(func, A_cpu, B_cpu, is_gpu=False)
    t_gpu = get_time(func, A_gpu, B_gpu, is_gpu=True)
    speedup = t_cpu / t_gpu  # Во сколько раз GPU быстрее
    print(f"{name:<28} | {t_cpu:<10.2f} | {t_gpu:<10.2f} | {speedup:.2f}x")

print("-" * 65)



