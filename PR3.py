import numpy as np
import matplotlib.pyplot as plt

# Константы для МЕДИ (Cu)
rho = 8900      # плотность, кг/м^3
c = 380         # удельная теплоемкость, Дж/(кг*К)
k = 400         # коэффициент теплопроводности, Вт/(м*К)
Q = 10000       # мощность внутренних источников тепла, Вт/м^3 (10 кВт/м³)

# Параметры задачи
Lx = 1.0        # длина по x, м
Ly = 0.5        # длина по y, м (ширина проводника)
total_time = 200  # время процесса, с

# Сетка
Nx = 31         # точек по x
Ny = 21         # точек по y
Nt = 1000       # шагов по времени

dx = Lx / (Nx - 1)  # шаг по x, м
dy = Ly / (Ny - 1)  # шаг по y, м
dt = total_time / Nt  # шаг по времени, с

# Коэффициенты
alpha = k / (rho * c)  # температуропроводность, м^2/с
coeff_x = alpha * dt / (dx**2)
coeff_y = alpha * dt / (dy**2)
source = dt * Q / (rho * c)  # вклад источников тепла, К

# Массив температур [x, y, t]
T = np.zeros((Nx, Ny, Nt))

# Начальные условия: везде 293К
T[:, :, 0] = 293.0

# Расчет
for j in range(0, Nt-1):
    # Копируем текущий слой
    T[:, :, j+1] = T[:, :, j].copy()

    # Расчет внутренних точек с источником тепла
    for i in range(1, Nx-1):
        for iy in range(1, Ny-1):
            T[i, iy, j+1] = T[i, iy, j] + \
                            coeff_x * (T[i+1, iy, j] - 2*T[i, iy, j] + T[i-1, iy, j]) + \
                            coeff_y * (T[i, iy+1, j] - 2*T[i, iy, j] + T[i, iy-1, j]) + \
                            source

    # Граничные условия: ВСЕ КРАЯ ХОЛОДНЫЕ (293К)
    T[0, :, j+1] = 293.0    # левый край
    T[-1, :, j+1] = 293.0   # правый край
    T[:, 0, j+1] = 293.0    # нижний край
    T[:, -1, j+1] = 293.0   # верхний край

# Создаем сетку для графиков
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')

# ГРАФИК 1: Распределение по длине в разные моменты (середина по ширине)
plt.figure(figsize=(10, 6))
mid_y = Ny // 2
times_to_plot = [0, Nt//4, Nt//2, 3*Nt//4, Nt-1]

for t_idx in times_to_plot:
    time_sec = t_idx * dt
    plt.plot(x, T[:, mid_y, t_idx], linewidth=2, label=f't={time_sec:.0f} с')
plt.xlabel('x, м')
plt.ylabel('Температура T, К')
plt.title('Нагрев изнутри (Qv=10 кВт/м³): распределение по длине')
plt.legend()
plt.grid(True)
plt.show()

# ГРАФИК 2: Изменение температуры во времени в разных точках
plt.figure(figsize=(10, 6))
time = np.linspace(0, total_time, Nt)
points_x = [0, Nx//4, Nx//2, 3*Nx//4, Nx-1]
labels_x = ['x=0 м (край)', 'x=0.25 м', 'x=0.5 м (центр)', 'x=0.75 м', 'x=1.0 м (край)']

for x_idx, label in zip(points_x, labels_x):
    plt.plot(time, T[x_idx, mid_y, :], linewidth=2, label=label)
plt.xlabel('Время t, с')
plt.ylabel('Температура T, К')
plt.title('Нагрев изнутри: изменение температуры в разных точках')
plt.legend()
plt.grid(True)
plt.show()

# ГРАФИК 3: 2D контурный график в конце процесса
plt.figure(figsize=(10, 6))
contour = plt.contourf(X, Y, T[:, :, -1], levels=20, cmap='hot')
plt.colorbar(contour, label='Температура, К')
plt.xlabel('x, м')
plt.ylabel('y, м')
plt.title(f'Нагрев изнутри: 2D поле в конце (t={total_time:.0f} с)')
plt.show()

# ГРАФИК 4: Профили по x при разных y
plt.figure(figsize=(10, 6))
y_positions = [0, Ny//4, Ny//2, 3*Ny//4, Ny-1]
for y_idx in y_positions:
    y_pos = y_idx * dy
    plt.plot(x, T[:, y_idx, -1], linewidth=2, label=f'y={y_pos:.2f} м')
plt.xlabel('x, м')
plt.ylabel('Температура, К')
plt.title('Нагрев изнутри: профили по длине (конец процесса)')
plt.legend()
plt.grid(True)
plt.show()

# ГРАФИК 5: Эволюция 2D поля (автоматический подбор цветов)
plt.figure(figsize=(15, 4))

# Находим общий мин и макс для всех трех графиков
T_min = min(T[:, :, 0].min(), T[:, :, Nt//2].min(), T[:, :, -1].min())
T_max = max(T[:, :, 0].max(), T[:, :, Nt//2].max(), T[:, :, -1].max())

plt.subplot(1, 3, 1)
im1 = plt.imshow(T[:, :, 0].T, origin='lower', extent=[0, Lx, 0, Ly],
                 aspect='auto', cmap='hot', vmin=T_min, vmax=T_max)
plt.colorbar(im1, label='T, К')
plt.xlabel('x, м')
plt.ylabel('y, м')
plt.title('t = 0 с')

plt.subplot(1, 3, 2)
im2 = plt.imshow(T[:, :, Nt//2].T, origin='lower', extent=[0, Lx, 0, Ly],
                 aspect='auto', cmap='hot', vmin=T_min, vmax=T_max)
plt.colorbar(im2, label='T, К')
plt.xlabel('x, м')
plt.ylabel('y, м')
plt.title(f't = {total_time//2:.0f} с')

plt.subplot(1, 3, 3)
im3 = plt.imshow(T[:, :, -1].T, origin='lower', extent=[0, Lx, 0, Ly],
                 aspect='auto', cmap='hot', vmin=T_min, vmax=T_max)
plt.colorbar(im3, label='T, К')
plt.xlabel('x, м')
plt.ylabel('y, м')
plt.title(f't = {total_time:.0f} с')

plt.suptitle('Эволюция 2D температурного поля (нагрев изнутри)')
plt.tight_layout()
plt.show()

print("=== 2D РАСЧЕТ: НАГРЕВ ИЗНУТРИ ===")
print(f"Материал: Медь (Cu)")
print(f"Внутренний источник: Q = {Q} Вт/м³ (10 кВт/м³)")
print(f"Граничные условия: ВСЕ КРАЯ = 293K")
print(f"\nРезультаты в конце процесса:")
print(f"  Макс температура: {T[:, :, -1].max():.1f} K (в центре)")
print(f"  Мин температура: {T[:, :, -1].min():.1f} K (на краях)")
print(f"  Температура в центре (x=0.5м, y=0.25м): {T[Nx//2, Ny//2, -1]:.1f} K")
print(f"  Диапазон температур: от {T_min:.1f} до {T_max:.1f} K")


# ГРАФИК 6: 3D поверхность температура + время
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection='3d')

# Берем температуру в центре по y для всех x и t
mid_y = Ny // 2
T_mid = T[:, mid_y, :]  # размерность (Nx, Nt)

# Создаем сетку для 3D
X_3d, T_3d = np.meshgrid(x, time, indexing='ij')


# Строим поверхность
surf = ax.plot_surface(X_3d, T_3d, T_mid, cmap='hot', linewidth=0, antialiased=True, alpha=0.8)

ax.set_xlabel('x, м')
ax.set_ylabel('Время t, с')
ax.set_zlabel('Температура T, К')
ax.set_title('3D: Зависимость температуры от координаты и времени\n(сечение по центру y=0.25м)')

fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Температура, К')
plt.show()


