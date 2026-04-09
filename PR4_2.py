import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Константы для МЕДИ (Cu)
rho = 8900      # плотность, кг/м^3
c = 380         # удельная теплоемкость, Дж/(кг*К)
k = 400         # коэффициент теплопроводности, Вт/(м*К)
Q_heater = 1e6  # мощность нагревателя, Вт/м³

# Параметры задачи
L = 2.0         # длина стержня, м (по x)
square_side = 1.0  # сторона квадратного сечения, м (по y и z)
total_time = 600   # время процесса, с

# ============ ПАРАМЕТРЫ НАГРЕВАТЕЛЕЙ ============
# Исходный прямоугольник (для расчета площади)
orig_width_y = 0.3   # по y, м (короткая сторона)
orig_width_z = 0.4   # по z, м (длинная сторона)
orig_area = orig_width_y * orig_width_z  # 0.12 м²

# Три нагревателя, суммарной площадью равной исходному
num_heaters = 3
heater_short = orig_width_y / num_heaters  # 0.1 м (короткая сторона)
heater_long = orig_width_z                # 0.4 м (длинная сторона)

# ============ КЛЮЧЕВЫЕ КОНСТАНТЫ ============
R = 0.35  # расстояние от центра стержня до центра каждого нагревателя, м

# Углы расположения нагревателей (в градусах)
# 0° = вверх (12 часов), 120°, 240°
angles_deg = [0, 120, 240]

# ============ ПОВОРОТ КАЖДОГО НАГРЕВАТЕЛЯ ВОКРУГ СВОЕЙ ОСИ ============
# Задай поворот для каждого нагревателя (в градусах)
# 0° = длинная сторона горизонтальна
# 90° = длинная сторона вертикальна
heater_rotations_deg = [90, -30, 30]
# Например: [0, 30, 60] или [90, 90, 90] и т.д.
# =================================================================

angles_rad = np.deg2rad(angles_deg)
rotations_rad = np.deg2rad(heater_rotations_deg)

# Сетка
Ny = 91         # точек по y
Nz = 91         # точек по z
Nt = 10000       # шагов по времени

dy = square_side / (Ny - 1)
dz = square_side / (Nz - 1)
dt = total_time / Nt

# Коэффициенты
alpha = k / (rho * c)
coeff_y = alpha * dt / (dy**2)
coeff_z = alpha * dt / (dz**2)
source = dt * Q_heater / (rho * c)

# Проверка устойчивости
print(f"coeff_y = {coeff_y:.4f}, coeff_z = {coeff_z:.4f}")
print(f"Сумма = {coeff_y + coeff_z:.4f} (должно быть <= 0.5 для устойчивости)")

# Координаты
y_coords = np.linspace(0, square_side, Ny)
z_coords = np.linspace(0, square_side, Nz)
Y, Z = np.meshgrid(y_coords, z_coords, indexing='ij')

# Центр стержня
center_y = square_side / 2
center_z = square_side / 2

# Функция для создания повернутого прямоугольника
def create_rotated_rectangle_mask(center_y, center_z, width_short, width_long, angle_rad):
    """
    Создает маску для прямоугольника, повернутого на угол angle_rad
    width_short - размер по локальной оси y (короткая сторона)
    width_long - размер по локальной оси z (длинная сторона)
    angle_rad - угол поворота (0 = длинная сторона горизонтальна)
    """
    mask = np.zeros((Ny, Nz), dtype=bool)

    half_short = width_short / 2
    half_long = width_long / 2

    # Углы прямоугольника в локальной системе
    corners_local = np.array([
        [-half_short, -half_long],
        [ half_short, -half_long],
        [ half_short,  half_long],
        [-half_short,  half_long]
    ])

    # Матрица поворота
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    # Поворачиваем углы и смещаем в центр нагревателя
    corners_rotated = corners_local @ rot_matrix.T
    corners_rotated[:, 0] += center_y
    corners_rotated[:, 1] += center_z

    # Находим bounding box для ускорения
    min_y = max(0, int(np.min(corners_rotated[:, 0]) / dy) - 1)
    max_y = min(Ny, int(np.max(corners_rotated[:, 0]) / dy) + 2)
    min_z = max(0, int(np.min(corners_rotated[:, 1]) / dz) - 1)
    max_z = min(Nz, int(np.max(corners_rotated[:, 1]) / dz) + 2)

    # Проверяем каждую точку в bounding box
    for iy in range(min_y, max_y):
        for iz in range(min_z, max_z):
            dx_local = y_coords[iy] - center_y
            dz_local = z_coords[iz] - center_z

            # Поворачиваем обратно
            x_local = dx_local * cos_a + dz_local * sin_a
            z_local = -dx_local * sin_a + dz_local * cos_a

            if abs(x_local) <= half_short and abs(z_local) <= half_long:
                mask[iy, iz] = True

    return mask

# Создаем маску для всех нагревателей
heater_mask = np.zeros((Ny, Nz), dtype=bool)

print(f"\n=== ТРИ НАГРЕВАТЕЛЯ ПО КРУГУ ===")
print(f"R = {R} м")
print(f"Углы положения: {angles_deg}°")
print(f"Повороты нагревателей: {heater_rotations_deg}°")

for i, (angle_rad, rot_rad) in enumerate(zip(angles_rad, rotations_rad)):
    # Центр нагревателя на окружности радиуса R
    heater_center_y = center_y + R * np.sin(angle_rad)
    heater_center_z = center_z + R * np.cos(angle_rad)

    # Создаем прямоугольник с заданным поворотом
    mask = create_rotated_rectangle_mask(heater_center_y, heater_center_z,
                                         heater_short, heater_long, rot_rad)
    heater_mask = heater_mask | mask

    print(f"Нагреватель {i+1}: положение={angles_deg[i]}°, поворот={heater_rotations_deg[i]}°, "
          f"центр=({heater_center_y:.3f}, {heater_center_z:.3f})")

# Площадь нагревателей
heater_area = np.sum(heater_mask) * dy * dz
print(f"\nИсходный нагреватель: {orig_width_y}x{orig_width_z} м, площадь {orig_area:.4f} м²")
print(f"Каждый нагреватель: {heater_short}x{heater_long} м")
print(f"Суммарная площадь нагревателей (расчетная): {heater_area:.4f} м²")

# Массив температур
T = np.zeros((Ny, Nz, Nt))
T[:, :, 0] = 293.0

# Расчет
print("\nНачало расчета...")
for j in range(0, Nt-1):
    for iy in range(1, Ny-1):
        for iz in range(1, Nz-1):
            q_source = source if heater_mask[iy, iz] else 0
            T[iy, iz, j+1] = T[iy, iz, j] + \
                             coeff_y * (T[iy+1, iz, j] - 2*T[iy, iz, j] + T[iy-1, iz, j]) + \
                             coeff_z * (T[iy, iz+1, j] - 2*T[iy, iz, j] + T[iy, iz-1, j]) + \
                             q_source

    T[0, :, j+1] = T[1, :, j+1]
    T[-1, :, j+1] = T[-2, :, j+1]
    T[:, 0, j+1] = T[:, 1, j+1]
    T[:, -1, j+1] = T[:, -2, j+1]

    if (j+1) % 500 == 0:
        print(f"Шаг {j+1}/{Nt} завершен")

print("Расчет завершен!")

# Координаты
y = np.linspace(0, square_side, Ny)
z = np.linspace(0, square_side, Nz)
time = np.linspace(0, total_time, Nt)

# ============ ПОДСЧЕТ ПОЛЕЗНОЙ ПЛОЩАДИ НАГРЕВА ============
T_final = T[:, :, -1]

if np.any(np.isnan(T_final)) or np.any(np.isinf(T_final)):
    print("ОШИБКА: В расчетах появились NaN или Inf! Уменьшите dt.")
    exit()

heated_above_310 = (T_final > 310) & (~heater_mask)
count_heated = np.sum(heated_above_310)

total_area = square_side * square_side
area_outside_heater = total_area - heater_area
area_per_point = dy * dz
heated_area = count_heated * area_per_point
percent_heated = (heated_area / area_outside_heater) * 100 if area_outside_heater > 0 else 0

print(f"\n=== ПОДСЧЕТ ПОЛЕЗНОЙ ПЛОЩАДИ НАГРЕВА ===")
print(f"Пороговая температура: 310 K")
print(f"R = {R} м")
print(f"Общая площадь сечения: {total_area:.4f} м²")
print(f"Площадь нагревателей (суммарная): {heater_area:.4f} м²")
print(f"Площадь вне нагревателей: {area_outside_heater:.4f} м²")
print(f"\nКоличество точек вне нагревателей с T > 310K: {count_heated}")
print(f"Площадь, нагретая выше 310K: {heated_area:.4f} м²")
print(f"Процент нагретой площади: {percent_heated:.1f}%")

# ============ ГРАФИК 1: Сечение (Y-Z) в начале, середине и конце ============
plt.figure(figsize=(15, 4))

T_vals = T[:, :, :]
T_min = np.percentile(T_vals, 1)
T_max = np.percentile(T_vals, 99)

plt.subplot(1, 3, 1)
im1 = plt.imshow(T[:, :, 0].T, origin='lower', extent=[0, square_side, 0, square_side],
                 aspect='equal', cmap='hot', vmin=T_min, vmax=T_max)
plt.colorbar(im1, label='Температура, К')
plt.xlabel('y, м')
plt.ylabel('z, м')
plt.title(f't = 0 с')

plt.subplot(1, 3, 2)
im2 = plt.imshow(T[:, :, Nt//2].T, origin='lower', extent=[0, square_side, 0, square_side],
                 aspect='equal', cmap='hot', vmin=T_min, vmax=T_max)
plt.colorbar(im2, label='Температура, К')
plt.xlabel('y, м')
plt.ylabel('z, м')
plt.title(f't = {total_time//2:.0f} с')

plt.subplot(1, 3, 3)
im3 = plt.imshow(T[:, :, -1].T, origin='lower', extent=[0, square_side, 0, square_side],
                 aspect='equal', cmap='hot', vmin=T_min, vmax=T_max)
plt.colorbar(im3, label='Температура, К')
plt.xlabel('y, м')
plt.ylabel('z, м')
plt.title(f't = {total_time:.0f} с')

plt.suptitle(f'Три нагревателя по кругу (R={R} м, повороты {heater_rotations_deg}°)')
plt.tight_layout()
plt.show()

# ============ ГРАФИК 2: Температура в разных точках ============
plt.figure(figsize=(12, 6))

center_y_idx = Ny // 2
center_z_idx = Nz // 2
temp_center = T[center_y_idx, center_z_idx, :]
temp_edge = T[0, 0, :]

# Находим точку внутри нагревателя
heater_points = np.where(heater_mask)
if len(heater_points[0]) > 0:
    temp_heater = T[heater_points[0][0], heater_points[1][0], :]
else:
    temp_heater = np.zeros_like(temp_center)

plt.plot(time, temp_center, linewidth=2, label='Центр стержня')
plt.plot(time, temp_heater, linewidth=2, label='Внутри нагревателя')
plt.plot(time, temp_edge, linewidth=2, label='Край стержня (угол)')
plt.axhline(y=310, color='r', linestyle='--', linewidth=1, label='Порог 310 K')
plt.xlabel('Время t, с')
plt.ylabel('Температура T, К')
plt.title(f'Три нагревателя (R={R} м, повороты {heater_rotations_deg}°)')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

# ============ ГРАФИК 3: Зоны нагрева выше 310K ============
plt.figure(figsize=(10, 8))

display_mask = np.zeros((Ny, Nz))
display_mask[heated_above_310] = 1
display_mask[heater_mask] = 2

cmap_custom = ListedColormap(['darkblue', 'yellow', 'red'])
plt.imshow(display_mask.T, origin='lower', extent=[0, square_side, 0, square_side],
           aspect='equal', cmap=cmap_custom, norm=plt.Normalize(0, 2))
plt.colorbar(ticks=[0.25, 1, 2], label='Зона',
             format=plt.FuncFormatter(lambda x, _: {0.25: 'Холодная (<310K)',
                                                    1: 'Нагретая (>310K)',
                                                    2: 'Нагреватель'}[x]))
plt.xlabel('y, м')
plt.ylabel('z, м')
plt.title(f'Три нагревателя (R={R} м, повороты {heater_rotations_deg}°)\nНагрето {percent_heated:.1f}% площади вне нагревателей')
plt.show()

print(f"\n=== ИТОГО ===")
print(f"При R = {R} м, повороты {heater_rotations_deg}°")
print(f"Полезная площадь нагрева = {heated_area:.4f} м² ({percent_heated:.1f}%)")