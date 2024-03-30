import numpy as np
from scipy.integrate import odeint
from scipy.linalg import eig
import matplotlib.pyplot as plt


# Функции
def solve_pendulum_system(m, L, L1, k, g, beta, phi1_0, phi2_0, dot_phi1_0, dot_phi2_0, t_max):
    def equations(y, t, m, L, L1, k, g, beta):
        phi1, z1, phi2, z2 = y
        f = [z1,
             (-beta * z1 - m * g * phi1 + k * (L1 / L) * (phi2 - phi1)) / (m * L),
             z2,
             (-beta * z2 - m * g * phi2 + k * (L1 / L) * (phi1 - phi2)) / (m * L)]
        return f

    initial_conditions = [phi1_0, dot_phi1_0, phi2_0, dot_phi2_0]
    t = np.linspace(0, t_max, 1000)
    sol = odeint(equations, initial_conditions, t, args=(m, L, L1, k, g, beta))
    return t, sol


def calculate_normal_frequencies(m, L, L1, k, g):
    A = np.array([[m * L ** 2, 0],
                  [0, m * L ** 2]])
    B = np.array([[k * L1 + m * g * L, -k * L1],
                  [-k * L1, k * L1 + m * g * L]])
    eigvals, _ = eig(B, A)
    normal_modes = np.sqrt(np.abs(eigvals.real))
    return normal_modes


# Параметры системы
m = 1.0  # масса маятника
L = 1.0  # длина подвеса
L1 = 0.5  # расстояние между маятниками, где прикреплена пружина
k = 10.0  # коэффициент жёсткости пружины
g = 9.8  # ускорение свободного падения
beta = 0.1  # коэффициент затухания
t_max = 20  # максимальное время для анализа

# Начальные условия
phi1_0 = 0.1  # начальное отклонение первого маятника
phi2_0 = -0.1  # начальное отклонение второго маятника
dot_phi1_0 = 0.0  # начальная угловая скорость первого маятника
dot_phi2_0 = 0.0  # начальная угловая скорость второго маятника

# Решение системы и построение графиков
t, sol = solve_pendulum_system(m, L, L1, k, g, beta, phi1_0, phi2_0, dot_phi1_0, dot_phi2_0, t_max)

plt.figure(figsize=(12, 10))

# Общие графики углов отклонения
plt.subplot(2, 2, 1)
plt.plot(t, sol[:, 0], label='Маятник 1', color='blue')
plt.plot(t, sol[:, 2], label='Маятник 2')
plt.title('Углы отклонения')
plt.xlabel('Время (с)')
plt.ylabel('Угол (рад)')
plt.legend()
plt.grid(True)

# Общие графики угловых скоростей
plt.subplot(2, 2, 2)
plt.plot(t, sol[:, 1], label='Маятник 1', color='blue')
plt.plot(t, sol[:, 3], label='Маятник 2')
plt.title('Угловые скорости')
plt.xlabel('Время (с)')
plt.ylabel('Скорость (рад/с)')
plt.legend()
plt.grid(True)

# Индивидуальные графики для маятника 1
plt.figure(figsize=(12, 5))  # Новое окно для отдельных графиков маятника 1

plt.subplot(1, 2, 1)
plt.plot(t, sol[:, 0], color='blue')
plt.title('Углы отклонения маятника 1')
plt.xlabel('Время (с)')
plt.ylabel('Угол (рад)')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(t, sol[:, 1], color='blue')
plt.title('Угловые скорости маятника 1')
plt.xlabel('Время (с)')
plt.ylabel('Скорость (рад/с)')
plt.grid(True)

# Индивидуальные графики для маятника 2
plt.figure(figsize=(12, 5))  # Новое окно для отдельных графиков маятника 2

plt.subplot(1, 2, 1)
plt.plot(t, sol[:, 2])
plt.title('Углы отклонения маятника 2')
plt.xlabel('Время (с)')
plt.ylabel('Угол (рад)')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(t, sol[:, 3])
plt.title('Угловые скорости маятника 2')
plt.xlabel('Время (с)')
plt.ylabel('Скорость (рад/с)')
plt.grid(True)

plt.tight_layout()

# Выводим информацию о нормальных частотах
normal_modes = calculate_normal_frequencies(m, L, L1, k, g)
print(f"Нормальные частоты: {normal_modes} рад/с")

# Показываем графики
plt.show()
