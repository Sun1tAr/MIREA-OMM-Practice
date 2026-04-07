import numpy as np
import matplotlib.pyplot as plt

# Константы для МЕДИ (Cu)
rho = 8900      # плотность, кг/м^3
c = 380         # удельная теплоемкость, Дж/(кг*К)
k = 400         # коэффициент теплопроводности, Вт/(м*К)
Q = 0           # мощность внутренних источников тепла, Вт/м^3

# Параметры задачи
Lx = 1.0        # длина по x, м
Ly = 0.5        # длина по y, м (ширина проводника)
total_time = 1800  # время процесса, с

# Сетка
Nx = 30         # точек по x
Ny = 20         # точек по y
Nt = 1000       # шагов по времени

dx = Lx / (Nx - 1)  # шаг по x, м
dy = Ly / (Ny - 1)  # шаг по y, м
dt = total_time / Nt  # шаг по времени, с

# Коэффициенты
alpha = k / (rho * c)  # температуропроводность, м^2/с
coeff_x = alpha * dt / (dx**2)
coeff_y = alpha * dt / (dy**2)

# Массив температур [x, y, t]
T = np.zeros((Nx, Ny, Nt))

# Начальные условия: везде 293К
T[:, :, 0] = 293.0

# Расчет
for j in range(0, Nt-1):
    # 1. Сначала копируем текущий слой на следующий
    T[:, :, j+1] = T[:, :, j].copy()

    # 2. Расчет внутренних точек (обновляем следующий слой)
    for i in range(1, Nx-1):
        for iy in range(1, Ny-1):
            T[i, iy, j+1] = T[i, iy, j] + \
                            coeff_x * (T[i+1, iy, j] - 2*T[i, iy, j] + T[i-1, iy, j]) + \
                            coeff_y * (T[i, iy+1, j] - 2*T[i, iy, j] + T[i, iy-1, j])

    # 3. Граничные условия ДИРИХЛЕ (заданная температура) на левом и правом краю
    T[0, :, j+1] = 393.0    # левый край горячий
    T[-1, :, j+1] = 293.0   # правый край холодный

    # 4. Граничные условия НЕЙМАНА (теплоизоляция) на верхнем и нижнем краю
    # Производная по y = 0 означает T[i, 0] = T[i, 1] и T[i, -1] = T[i, -2]
    for i in range(Nx):
        T[i, 0, j+1] = T[i, 1, j+1]      # нижний край (y=0)
        T[i, -1, j+1] = T[i, -2, j+1]    # верхний край (y=Ly)

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
plt.title('Распределение температуры по длине (середина по ширине)')
plt.legend()
plt.grid(True)
plt.show()

# ГРАФИК 2: Изменение температуры во времени в разных точках
plt.figure(figsize=(10, 6))
time = np.linspace(0, total_time, Nt)
points_x = [0, Nx//4, Nx//2, 3*Nx//4, Nx-1]
labels_x = ['x=0 м (горячий край)', 'x=0.25 м', 'x=0.5 м', 'x=0.75 м', 'x=1.0 м (холодный край)']

for x_idx, label in zip(points_x, labels_x):
    plt.plot(time, T[x_idx, mid_y, :], linewidth=2, label=label)
plt.xlabel('Время t, с')
plt.ylabel('Температура T, К')
plt.title('Изменение температуры в разных точках (середина по ширине)')
plt.legend()
plt.grid(True)
plt.show()

# ГРАФИК 3: 2D контурный график в конце процесса
plt.figure(figsize=(10, 6))
contour = plt.contourf(X, Y, T[:, :, -1], levels=20, cmap='hot')
plt.colorbar(contour, label='Температура, К')
plt.xlabel('x, м')
plt.ylabel('y, м')
plt.title(f'2D распределение температуры в конце (t={total_time:.0f} с)')
plt.show()

# ГРАФИК 4: Профили по ширине при разных x
plt.figure(figsize=(10, 6))
x_positions = [0, Nx//4, Nx//2, 3*Nx//4]
for x_idx in x_positions:
    x_pos = x_idx * dx
    plt.plot(y, T[x_idx, :, -1], 'o-', linewidth=2, label=f'x={x_pos:.2f} м')
plt.xlabel('y, м')
plt.ylabel('Температура, К')
plt.title('Профили температуры по ширине (конец процесса)')
plt.legend()
plt.grid(True)
plt.show()

# ГРАФИК 5: 2D поля в начале, середине и конце (вместо 3D)
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.imshow(T[:, :, 0].T, origin='lower', extent=[0, Lx, 0, Ly],
           aspect='auto', cmap='hot', vmin=290, vmax=400)
plt.colorbar(label='T, К')
plt.xlabel('x, м')
plt.ylabel('y, м')
plt.title('t = 0 с')

plt.subplot(1, 3, 2)
plt.imshow(T[:, :, Nt//2].T, origin='lower', extent=[0, Lx, 0, Ly],
           aspect='auto', cmap='hot', vmin=290, vmax=400)
plt.colorbar(label='T, К')
plt.xlabel('x, м')
plt.ylabel('y, м')
plt.title(f't = {total_time//2:.0f} с')

plt.subplot(1, 3, 3)
plt.imshow(T[:, :, -1].T, origin='lower', extent=[0, Lx, 0, Ly],
           aspect='auto', cmap='hot', vmin=290, vmax=400)
plt.colorbar(label='T, К')
plt.xlabel('x, м')
plt.ylabel('y, м')
plt.title(f't = {total_time:.0f} с')

plt.suptitle('Эволюция 2D температурного поля')
plt.tight_layout()
plt.show()

print("=== 2D РАСЧЕТ ДЛЯ МЕДНОГО ПРОВОДНИКА ===")
print(f"Материал: Медь (Cu)")
print(f"Граничные условия:")
print(f"  x=0: T=393K (нагрев)")
print(f"  x={Lx}: T=293K (холод)")
print(f"  y=0 и y={Ly}: теплоизоляция (dT/dy=0)")
print(f"\nРезультаты в центре (x=0.5м, y=0.25м):")
print(f"  Начало: {T[Nx//2, Ny//2, 0]:.1f} K")
print(f"  Середина: {T[Nx//2, Ny//2, Nt//2]:.1f} K")
print(f"  Конец: {T[Nx//2, Ny//2, -1]:.1f} K")