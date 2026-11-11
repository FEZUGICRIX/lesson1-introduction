import torch
import math

# ==============================================================================
# ЗАДАНИЕ 2.1: Функция f(x, y, z) = x^2 + y^2 + z^2 + 2*x*y*z
# ==============================================================================
print("=" * 60)
print("2.1 Градиенты функции f(x, y, z)")
print("=" * 60)

# Создаем переменные и включаем отслеживание градиентов
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
z = torch.tensor(4.0, requires_grad=True)

# Считаем функцию
f = x**2 + y**2 + z**2 + 2 * x * y * z

# Вычисляем градиенты
f.backward()

# Аналитические формулы для проверки
grad_x_formula = 2 * 2.0 + 2 * 3.0 * 4.0  # 2x + 2yz = 28
grad_y_formula = 2 * 3.0 + 2 * 2.0 * 4.0  # 2y + 2xz = 22
grad_z_formula = 2 * 4.0 + 2 * 2.0 * 3.0  # 2z + 2xy = 20

print(f"df/dx: PyTorch = {x.grad.item():.1f} | По формуле = {grad_x_formula:.1f}")
print(f"df/dy: PyTorch = {y.grad.item():.1f} | По формуле = {grad_y_formula:.1f}")
print(f"df/dz: PyTorch = {z.grad.item():.1f} | По формуле = {grad_z_formula:.1f}")


# ==============================================================================
# ЗАДАНИЕ 2.2: Градиент функции потерь MSE (Linear Regression)
# ==============================================================================
print("\n" + "=" * 60)
print("2.2 Градиенты ошибки MSE для y = w * x + b")
print("=" * 60)

# Данные: вход x и правильный ответ y_true (для y = 2x + 1)
x_data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
y_true = torch.tensor([3.0, 5.0, 7.0, 9.0, 11.0])

# Настраиваемые параметры модели (ручки w и b)
w = torch.tensor(1.00, requires_grad=True)
b = torch.tensor(0.00, requires_grad=True)

# Прогноз модели: y_pred = w * x + b
y_pred = w * x_data + b

# Ошибка MSE: среднее от (прогноз - факт)^2
n = len(x_data)
mse = (1.0 / n) * torch.sum((y_pred - y_true) ** 2)

# Считаем градиенты ошибки по w и b
mse.backward()

# Проверка по формуле
diff = y_pred.detach() - y_true
grad_w_formula = (2.0 / n) * torch.sum(diff * x_data).item()
grad_b_formula = (2.0 / n) * torch.sum(diff).item()

print(f"Ошибка MSE = {mse.item():.4f}")
print(f"Градиент по w: PyTorch = {w.grad.item():.4f} | По формуле = {grad_w_formula:.4f}")
print(f"Градиент по b: PyTorch = {b.grad.item():.4f} | По формуле = {grad_b_formula:.4f}")


# ==============================================================================
# ЗАДАНИЕ 2.3: Сложная функция f(x) = sin(x^2 + 1)
# ==============================================================================
print("\n" + "=" * 60)
print("2.3 Цепное правило для f(x) = sin(x^2 + 1)")
print("=" * 60)

x_val = 1.5
x = torch.tensor(x_val, requires_grad=True)
f = torch.sin(x**2 + 1)

# 1) Находим градиент через torch.autograd.grad (по требованию задачи)
grad_autograd = torch.autograd.grad(f, x)[0].item()


# 2) Аналитический расчёт по цепному правилу: df/dx = 2x * cos(x^2 + 1)
grad_formula = 2.0 * x_val * math.cos(x_val**2 + 1)

print(f"Точка x = {x_val}")
print(f"Через torch.autograd.grad: {grad_autograd:.6f}")
print(f"По формуле 2x*cos(x^2+1):   {grad_formula:.6f}")
print("Результаты полностью совпадают!")
