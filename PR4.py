import numpy as np
import matplotlib.pyplot as plt

# Константы для МЕДИ (Cu)
rho = 8900      # плотность, кг/м^3
c = 380         # удельная теплоемкость, Дж/(кг*К)
k = 400         # коэффициент теплопроводности, Вт/(м*К)
Q_heater = 1e6  # мощность нагревателя, Вт/м³

# Параметры задачи
L = 2.0         # длина стержня, м (по x)
square_side = 1.0  # сторона квадратного сечения, м (по y и z)
total_time = 600   # время процесса, с

# Размеры нагревателя в сечении (прямоугольник в Y-Z)
heater_width_y = 0.3   # по y, м
heater_width_z = 0.4   # по z, м

# Сетка (только сечение Y-Z, x фиксирован в центре стержня)
Ny = 41         # точек по y
Nz = 41         # точек по z
Nt = 1000       # шагов по времени

dy = square_side / (Ny - 1)
dz = square_side / (Nz - 1)
dt = total_time / Nt

# Коэффициенты
alpha = k / (rho * c)
coeff_y = alpha * dt / (dy**2)
coeff_z = alpha * dt / (dz**2)
source = dt * Q_heater / (rho * c)

# Массив температур сечения [y, z, t]
T = np.zeros((Ny, Nz, Nt))
T[:, :, 0] = 293.0

# Определяем область нагревателя в сечении (центр совпадает с центром стержня)
heater_y_start = Ny//2 - int(heater_width_y / dy / 2)
heater_y_end = Ny//2 + int(heater_width_y / dy / 2)
heater_z_start = Nz//2 - int(heater_width_z / dz / 2)
heater_z_end = Nz//2 + int(heater_width_z / dz / 2)

# Маска нагревателя (True - внутри нагревателя, False - снаружи)
heater_mask = np.zeros((Ny, Nz), dtype=bool)
heater_mask[heater_y_start:heater_y_end, heater_z_start:heater_z_end] = True

print(f"Нагреватель в сечении: {heater_width_y}x{heater_width_z} м")
print(f"Центр стержня: y=0.5м, z=0.5м")
print(f"Нагреватель занимает {np.sum(heater_mask)} точек из {Ny*Nz}")

# Расчет
for j in range(0, Nt-1):
    T[:, :, j+1] = T[:, :, j].copy()

    # Расчет внутренних точек
    for iy in range(1, Ny-1):
        for iz in range(1, Nz-1):
            # Добавляем источник только внутри нагревателя
            q_source = source if heater_mask[iy, iz] else 0
            T[iy, iz, j+1] = T[iy, iz, j] + \
                             coeff_y * (T[iy+1, iz, j] - 2*T[iy, iz, j] + T[iy-1, iz, j]) + \
                             coeff_z * (T[iy, iz+1, j] - 2*T[iy, iz, j] + T[iy, iz-1, j]) + \
                             q_source

    # Граничные условия: теплоизоляция на всех краях сечения
    T[0, :, j+1] = T[1, :, j+1]        # нижний край (y=0)
    T[-1, :, j+1] = T[-2, :, j+1]      # верхний край (y=1)
    T[:, 0, j+1] = T[:, 1, j+1]        # левый край (z=0)
    T[:, -1, j+1] = T[:, -2, j+1]      # правый край (z=1)

# Координаты для графиков
y = np.linspace(0, square_side, Ny)
z = np.linspace(0, square_side, Nz)
time = np.linspace(0, total_time, Nt)

# ============ ПОДСЧЕТ ПОЛЕЗНОЙ ПЛОЩАДИ НАГРЕВА ============
# Берем последний момент времени
T_final = T[:, :, -1]

# Точки, нагретые выше 310K (исключая нагреватель)
heated_above_310 = (T_final > 310) & (~heater_mask)

# Количество таких точек
count_heated = np.sum(heated_above_310)

# Общая площадь сечения (исключая нагреватель)
total_area_outside_heater = square_side * square_side - (heater_width_y * heater_width_z)
area_per_point = dy * dz
heated_area = count_heated * area_per_point
percent_heated = (heated_area / total_area_outside_heater) * 100

print(f"\n=== ПОДСЧЕТ ПОЛЕЗНОЙ ПЛОЩАДИ НАГРЕВА ===")
print(f"Пороговая температура: 310 K")
print(f"Общая площадь сечения: {square_side * square_side:.2f} м²")
print(f"Площадь нагревателя: {heater_width_y * heater_width_z:.2f} м²")
print(f"Площадь вне нагревателя: {total_area_outside_heater:.2f} м²")
print(f"\nКоличество точек вне нагревателя: {Ny*Nz - np.sum(heater_mask)}")
print(f"Количество точек вне нагревателя с T > 310K: {count_heated}")
print(f"Площадь, нагретая выше 310K (вне нагревателя): {heated_area:.4f} м²")
print(f"Процент нагретой площади (вне нагревателя): {percent_heated:.1f}%")

# ============ ГРАФИК 1: Сечение (Y-Z) в начале, середине и конце ============
plt.figure(figsize=(15, 4))

# Находим общий мин и макс для всех трех графиков
T_min = min(T[:, :, 0].min(), T[:, :, Nt//2].min(), T[:, :, -1].min())
T_max = max(T[:, :, 0].max(), T[:, :, Nt//2].max(), T[:, :, -1].max())

# Начало (t=0)
plt.subplot(1, 3, 1)
im1 = plt.imshow(T[:, :, 0].T, origin='lower', extent=[0, square_side, 0, square_side],
                 aspect='equal', cmap='hot', vmin=T_min, vmax=T_max)
plt.colorbar(im1, label='Температура, К')
plt.xlabel('y, м')
plt.ylabel('z, м')
plt.title(f'Сечение стержня, t = 0 с')

# Середина (t = total_time/2)
plt.subplot(1, 3, 2)
im2 = plt.imshow(T[:, :, Nt//2].T, origin='lower', extent=[0, square_side, 0, square_side],
                 aspect='equal', cmap='hot', vmin=T_min, vmax=T_max)
plt.colorbar(im2, label='Температура, К')
plt.xlabel('y, м')
plt.ylabel('z, м')
plt.title(f'Сечение стержня, t = {total_time//2:.0f} с')

# Конец (t = total_time)
plt.subplot(1, 3, 3)
im3 = plt.imshow(T[:, :, -1].T, origin='lower', extent=[0, square_side, 0, square_side],
                 aspect='equal', cmap='hot', vmin=T_min, vmax=T_max)
plt.colorbar(im3, label='Температура, К')
plt.xlabel('y, м')
plt.ylabel('z, м')
plt.title(f'Сечение стержня, t = {total_time:.0f} с')

plt.suptitle(f'Поперечное сечение стержня (тепловой градиент, Q={Q_heater/1e6:.0f} МВт/м³)')
plt.tight_layout()
plt.show()

# ============ ГРАФИК 2: Температура в центре нагревателя и на краю ============
plt.figure(figsize=(12, 6))

center_y = Ny//2
center_z = Nz//2
temp_center = T[center_y, center_z, :]
temp_edge = T[0, 0, :]

plt.plot(time, temp_center, linewidth=2, label='Центр нагревателя')
plt.plot(time, temp_edge, linewidth=2, label='Край стержня (угол)')
plt.axhline(y=310, color='r', linestyle='--', linewidth=1, label='Порог 310 K')
plt.xlabel('Время t, с')
plt.ylabel('Температура T, К')
plt.title('Изменение температуры в центре нагревателя и на краю стержня')
plt.legend()
plt.grid(True)
plt.show()

# ============ ГРАФИК 3: Зоны нагрева выше 310K (исключая нагреватель) ============
plt.figure(figsize=(10, 8))

# Создаем маску для отображения
display_mask = np.zeros((Ny, Nz))
display_mask[heated_above_310] = 1  # Нагретые зоны
display_mask[heater_mask] = 2       # Нагреватель (другой цвет)

# Создаем цветную карту
from matplotlib.colors import ListedColormap
cmap_custom = ListedColormap(['darkblue', 'yellow', 'red'])
bounds = [0, 0.5, 1.5, 2.5]
norm = plt.Normalize(0, 2)

plt.imshow(display_mask.T, origin='lower', extent=[0, square_side, 0, square_side],
           aspect='equal', cmap=cmap_custom, norm=norm)
plt.colorbar(ticks=[0.25, 1, 2], label='Зона',
             format=plt.FuncFormatter(lambda x, _: {0.25: 'Холодная (<310K)',
                                                    1: 'Нагретая (>310K)',
                                                    2: 'Нагреватель'}[x]))
plt.xlabel('y, м')
plt.ylabel('z, м')
plt.title(f'Зоны нагрева выше 310K (исключая нагреватель)\nНагрето {percent_heated:.1f}% площади вне нагревателя')
plt.show()

# ============ ДОПОЛНИТЕЛЬНО: профиль температуры по диагонали ============
plt.figure(figsize=(10, 6))

diag_points = min(Ny, Nz)
diag_temp = np.array([T[i, i, -1] for i in range(diag_points)])
diag_dist = np.sqrt(2) * np.linspace(0, square_side/2, diag_points)

plt.plot(diag_dist, diag_temp, 'o-', linewidth=2)
plt.axhline(y=310, color='r', linestyle='--', linewidth=1, label='Порог 310 K')
plt.xlabel('Расстояние от угла до центра, м')
plt.ylabel('Температура T, К')
plt.title('Профиль температуры по диагонали от угла до центра (конец процесса)')
plt.legend()
plt.grid(True)
plt.show()