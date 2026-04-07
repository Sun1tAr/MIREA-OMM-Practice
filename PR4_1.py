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

# Размеры исходного прямоугольного нагревателя
orig_width_y = 0.3   # по y, м (короткая сторона)
orig_width_z = 0.4   # по z, м (длинная сторона)

# После разрезания пополам по короткой стороне
heater_width_y = orig_width_y / 2  # 0.15 м
heater_width_z = orig_width_z      # 0.4 м
gap = 0.6             # расстояние между нагревателями = 0.3 м

# Сетка (уменьшил размер для устойчивости)
Ny = 51         # точек по y
Nz = 51         # точек по z
Nt = 2000       # шагов по времени (увеличил для устойчивости)

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
if coeff_y + coeff_z > 0.5:
    print("ВНИМАНИЕ! Схема может быть неустойчивой. Уменьшите dt или увеличьте dx/dy.")

# Координаты
y_coords = np.linspace(0, square_side, Ny)
z_coords = np.linspace(0, square_side, Nz)
Y, Z = np.meshgrid(y_coords, z_coords, indexing='ij')

# Центр стержня
center_y = square_side / 2
center_z = square_side / 2

# Позиции двух нагревателей
offset = gap / 2  # 0.15 м

# Левый нагреватель (смещен вниз по y)
heater1_y_start = center_y - offset - heater_width_y/2
heater1_y_end = center_y - offset + heater_width_y/2
# Правый нагреватель (смещен вверх по y)
heater2_y_start = center_y + offset - heater_width_y/2
heater2_y_end = center_y + offset + heater_width_y/2

# По z оба нагревателя центрированы
heater_z_start = center_z - heater_width_z/2
heater_z_end = center_z + heater_width_z/2

# Маска нагревателей
heater_mask = np.zeros((Ny, Nz), dtype=bool)

# Заполняем маску для первого нагревателя
y1_start_idx = max(0, int(heater1_y_start / dy))
y1_end_idx = min(Ny, int(heater1_y_end / dy) + 1)
z_start_idx = max(0, int(heater_z_start / dz))
z_end_idx = min(Nz, int(heater_z_end / dz) + 1)
heater_mask[y1_start_idx:y1_end_idx, z_start_idx:z_end_idx] = True

# Заполняем маску для второго нагревателя
y2_start_idx = max(0, int(heater2_y_start / dy))
y2_end_idx = min(Ny, int(heater2_y_end / dy) + 1)
heater_mask[y2_start_idx:y2_end_idx, z_start_idx:z_end_idx] = True

# Массив температур
T = np.zeros((Ny, Nz, Nt))
T[:, :, 0] = 293.0

print(f"\n=== ДВА НАГРЕВАТЕЛЯ (РАЗНЕСЕННЫЕ) ===")
print(f"Исходный нагреватель: {orig_width_y}x{orig_width_z} м")
print(f"После разрезания: {heater_width_y}x{heater_width_z} м каждый")
print(f"Расстояние между нагревателями: {gap} м")
print(f"Позиции нагревателей по y: {heater1_y_start:.3f}-{heater1_y_end:.3f} и {heater2_y_start:.3f}-{heater2_y_end:.3f}")
print(f"Позиции по z: {heater_z_start:.3f}-{heater_z_end:.3f}")
print(f"Суммарная площадь нагревателей: {np.sum(heater_mask) * dy * dz:.4f} м²")
print(f"Площадь исходного: {orig_width_y * orig_width_z:.4f} м²")

# Расчет
for j in range(0, Nt-1):
    # Расчет внутренних точек
    for iy in range(1, Ny-1):
        for iz in range(1, Nz-1):
            q_source = source if heater_mask[iy, iz] else 0
            T[iy, iz, j+1] = T[iy, iz, j] + \
                             coeff_y * (T[iy+1, iz, j] - 2*T[iy, iz, j] + T[iy-1, iz, j]) + \
                             coeff_z * (T[iy, iz+1, j] - 2*T[iy, iz, j] + T[iy, iz-1, j]) + \
                             q_source

    # Граничные условия: теплоизоляция
    T[0, :, j+1] = T[1, :, j+1]
    T[-1, :, j+1] = T[-2, :, j+1]
    T[:, 0, j+1] = T[:, 1, j+1]
    T[:, -1, j+1] = T[:, -2, j+1]

    # Вывод прогресса
    if (j+1) % 500 == 0:
        print(f"Шаг {j+1}/{Nt} завершен")

# Координаты для графиков
y = np.linspace(0, square_side, Ny)
z = np.linspace(0, square_side, Nz)
time = np.linspace(0, total_time, Nt)

# ============ ПОДСЧЕТ ПОЛЕЗНОЙ ПЛОЩАДИ НАГРЕВА ============
T_final = T[:, :, -1]

# Проверка на NaN и Inf
if np.any(np.isnan(T_final)) or np.any(np.isinf(T_final)):
    print("ОШИБКА: В расчетах появились NaN или Inf! Уменьшите dt.")
    exit()

# Точки, нагретые выше 310K (исключая нагреватели)
heated_above_310 = (T_final > 310) & (~heater_mask)

count_heated = np.sum(heated_above_310)

total_area = square_side * square_side
heater_area = np.sum(heater_mask) * dy * dz
area_outside_heater = total_area - heater_area
area_per_point = dy * dz
heated_area = count_heated * area_per_point
percent_heated = (heated_area / area_outside_heater) * 100 if area_outside_heater > 0 else 0

print(f"\n=== ПОДСЧЕТ ПОЛЕЗНОЙ ПЛОЩАДИ НАГРЕВА (ДВА НАГРЕВАТЕЛЯ) ===")
print(f"Пороговая температура: 310 K")
print(f"Общая площадь сечения: {total_area:.4f} м²")
print(f"Площадь нагревателей (суммарная): {heater_area:.4f} м²")
print(f"Площадь вне нагревателей: {area_outside_heater:.4f} м²")
print(f"\nКоличество точек вне нагревателей с T > 310K: {count_heated}")
print(f"Площадь, нагретая выше 310K (вне нагревателей): {heated_area:.4f} м²")
print(f"Процент нагретой площади (вне нагревателей): {percent_heated:.1f}%")

# ============ ГРАФИК 1: Сечение (Y-Z) в начале, середине и конце ============
plt.figure(figsize=(15, 4))

# Находим реальные мин и макс (игнорируем возможные выбросы)
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

plt.suptitle(f'Два нагревателя ({heater_width_y}x{heater_width_z} м, расстояние {gap} м)')
plt.tight_layout()
plt.show()

# ============ ГРАФИК 2: Температура в разных точках ============
plt.figure(figsize=(12, 6))

center_y_idx = Ny // 2
center_z_idx = Nz // 2
temp_center = T[center_y_idx, center_z_idx, :]  # между нагревателями
temp_edge = T[0, 0, :]
temp_heater = T[y1_start_idx + (y1_end_idx-y1_start_idx)//2, z_start_idx + (z_end_idx-z_start_idx)//2, :]

plt.plot(time, temp_center, linewidth=2, label='Центр стержня (между нагревателями)')
plt.plot(time, temp_heater, linewidth=2, label='Внутри нагревателя')
plt.plot(time, temp_edge, linewidth=2, label='Край стержня (угол)')
plt.axhline(y=310, color='r', linestyle='--', linewidth=1, label='Порог 310 K')
plt.xlabel('Время t, с')
plt.ylabel('Температура T, К')
plt.title('Два нагревателя: изменение температуры в разных точках')
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
plt.title(f'Два нагревателя: зоны нагрева выше 310K\nНагрето {percent_heated:.1f}% площади вне нагревателей')
plt.show()

# ============ ГРАФИК 4: Профиль температуры по y ============
plt.figure(figsize=(10, 6))

z_center_idx = Nz // 2
profile_y = T[:, z_center_idx, -1]

plt.plot(y, profile_y, linewidth=2)
plt.axhline(y=310, color='r', linestyle='--', linewidth=1, label='Порог 310 K')
plt.axvline(x=heater1_y_start, color='g', linestyle='--', linewidth=1, alpha=0.7)
plt.axvline(x=heater1_y_end, color='g', linestyle='--', linewidth=1, alpha=0.7)
plt.axvline(x=heater2_y_start, color='g', linestyle='--', linewidth=1, alpha=0.7)
plt.axvline(x=heater2_y_end, color='g', linestyle='--', linewidth=1, alpha=0.7, label='Границы нагревателей')
plt.xlabel('y, м')
plt.ylabel('Температура T, К')
plt.title('Профиль температуры по y (при z=0.5м, конец процесса)')
plt.legend()
plt.grid(True)
plt.show()

print("\nРасчет успешно завершен!")



