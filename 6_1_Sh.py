import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

# ============ СВОЙСТВА МАТЕРИАЛОВ ============
# Графит (C)
rho_C = 2250      # кг/м³
cp_C = 712        # Дж/(кг·К)
lambda_C = 1.59   # Вт/(м·К)

# Сухой песок (SiO2)
rho_SiO2 = 1520   # кг/м³
cp_SiO2 = 835     # Дж/(кг·К)
lambda_SiO2 = 0.3 # Вт/(м·К)

# Вода (H2O)
rho_H2O = 998.2   # кг/м³
cp_H2O = 4182     # Дж/(кг·К)
lambda_H2O = 0.56 # Вт/(м·К)

# ============ МАССОВЫЕ ДОЛИ ============
w_C = 0.4
w_SiO2 = 0.5
w_H2O = 0.1

# ============ РАСЧЁТ СВОЙСТВ СМЕСИ ============
rho_mix = 1 / (w_C/rho_C + w_SiO2/rho_SiO2 + w_H2O/rho_H2O)
cp_mix = w_C * cp_C + w_SiO2 * cp_SiO2 + w_H2O * cp_H2O
lambda_mix = (lambda_H2O * w_H2O / rho_H2O +
              lambda_SiO2 * w_SiO2 / rho_SiO2 +
              lambda_C * w_C / rho_C) * rho_mix

print(f"=== СВОЙСТВА СМЕСИ ===")
print(f"Плотность: {rho_mix:.2f} кг/м³")
print(f"Теплоёмкость: {cp_mix:.2f} Дж/(кг·К)")
print(f"Теплопроводность: {lambda_mix:.4f} Вт/(м·К)")

# ============ ПАРАМЕТРЫ НАГРЕВАТЕЛЯ ============
width_y = 0.3      # м
width_z = 0.4      # м
Q_heater = 5e7     # 50 МВт/м³

# Свойства нагревателя (медь)
rho_heater = 8900
cp_heater = 380
lambda_heater = 400

# ============ ПАРАМЕТРЫ СЕТКИ ============
total_time = 600    # с
square_side = 1.0   # м
Ny, Nz = 61, 61
Nt = 4000

dy = square_side / (Ny - 1)
dz = square_side / (Nz - 1)
dt = total_time / Nt

print(f"\n=== ПАРАМЕТРЫ РАСЧЁТА ===")
print(f"Размер сетки: {Ny}x{Nz}")
print(f"dy = {dy:.5f} м, dz = {dz:.5f} м")
print(f"dt = {dt:.4f} с, Nt = {Nt}")
print(f"Общее время: {total_time} с")

# ============ ИНИЦИАЛИЗАЦИЯ ПОЛЕЙ ============
rho_map = np.full((Ny, Nz), rho_mix)
cp_map = np.full((Ny, Nz), cp_mix)
lambda_map = np.full((Ny, Nz), lambda_mix)

# ============ ПРЯМОУГОЛЬНЫЙ НАГРЕВАТЕЛЬ ============
center_y = square_side / 2
center_z = square_side / 2
y_coords = np.linspace(0, square_side, Ny)
z_coords = np.linspace(0, square_side, Nz)
Y, Z = np.meshgrid(y_coords, z_coords, indexing='ij')

heater_mask = (Y >= center_y - width_y/2) & (Y <= center_y + width_y/2) & \
              (Z >= center_z - width_z/2) & (Z <= center_z + width_z/2)

# Замена материала в зоне нагревателя
rho_map[heater_mask] = rho_heater
cp_map[heater_mask] = cp_heater
lambda_map[heater_mask] = lambda_heater

# ============ ПАРАМЕТРЫ РАСЧЕТА ============
epsilon = 1e-10
source_val = dt * Q_heater / (rho_map * cp_map + epsilon)
coeff_y_map = lambda_map * dt / ((rho_map * cp_map + epsilon) * dy**2)
coeff_z_map = lambda_map * dt / ((rho_map * cp_map + epsilon) * dz**2)

# Массив температур
T = np.zeros((Ny, Nz, Nt))
T[:, :, 0] = 293.0

# ============ РАСЧЕТ ============
print("\nНачало расчета...")

for j in range(0, Nt-1):
    for iy in range(1, Ny-1):
        for iz in range(1, Nz-1):
            q_source = source_val[iy, iz] if heater_mask[iy, iz] else 0
            diffusion = coeff_y_map[iy, iz] * (T[iy+1, iz, j] - 2*T[iy, iz, j] + T[iy-1, iz, j]) + \
                        coeff_z_map[iy, iz] * (T[iy, iz+1, j] - 2*T[iy, iz, j] + T[iy, iz-1, j])
            T[iy, iz, j+1] = T[iy, iz, j] + diffusion + q_source

    # Граничные условия: теплоизоляция
    T[0, :, j+1] = T[1, :, j+1]
    T[-1, :, j+1] = T[-2, :, j+1]
    T[:, 0, j+1] = T[:, 1, j+1]
    T[:, -1, j+1] = T[:, -2, j+1]

    if (j+1) % 500 == 0:
        print(f"Шаг {j+1}/{Nt} | T_max = {np.max(T[:, :, j+1]):.1f}K")

print("Расчет завершен!")

# ============ ГРАФИКИ ============
y = np.linspace(0, square_side, Ny)
z = np.linspace(0, square_side, Nz)

T_final = T[:, :, -1]

# ============ ГРАФИК 1: Градиент температуры ============
plt.figure(figsize=(10, 8))

T_min = np.percentile(T_final, 1)
T_max = np.percentile(T_final, 99)

im_temp = plt.imshow(T_final.T, origin='lower', extent=[0, square_side, 0, square_side],
                     aspect='equal', cmap='hot', vmin=T_min, vmax=T_max)
plt.colorbar(im_temp, label='Температура, К')
plt.xlabel('y, м')
plt.ylabel('z, м')
plt.title(f'Распределение температуры в конце процесса (t={total_time:.0f} с)\n'
          f'Смесь: графит {w_C*100:.0f}% / песок {w_SiO2*100:.0f}% / вода {w_H2O*100:.0f}%')

# Рисуем контур нагревателя (создаём новый объект)
heater_rect1 = Rectangle((center_y - width_y/2, center_z - width_z/2),
                         width_y, width_z,
                         edgecolor='cyan', facecolor='none', linewidth=2)
plt.gca().add_patch(heater_rect1)

plt.show()

# ============ ГРАФИК 2: Область полезного нагрева ============
plt.figure(figsize=(10, 8))

heated_above_310 = (T_final > 310) & (~heater_mask)
display_mask = np.zeros((Ny, Nz))
display_mask[heated_above_310] = 1
display_mask[heater_mask] = 2

cmap_custom = ListedColormap(['darkblue', 'yellow', 'red'])
im_zones = plt.imshow(display_mask.T, origin='lower', extent=[0, square_side, 0, square_side],
                      aspect='equal', cmap=cmap_custom, norm=plt.Normalize(0, 2))
plt.colorbar(im_zones, ticks=[0.25, 1, 2], label='Зона',
             format=plt.FuncFormatter(lambda x, _: {0.25: 'Холодная (<310K)',
                                                    1: 'Нагретая (>310K)',
                                                    2: 'Нагреватель'}[x]))
plt.xlabel('y, м')
plt.ylabel('z, м')

# Рисуем контур нагревателя (новый объект)
heater_rect2 = Rectangle((center_y - width_y/2, center_z - width_z/2),
                         width_y, width_z,
                         edgecolor='white', facecolor='none', linewidth=2)
plt.gca().add_patch(heater_rect2)

# Подсчёт процента полезной площади
heater_area = np.sum(heater_mask) * dy * dz
heated_area = np.sum(heated_above_310) * dy * dz
area_outside = 1.0 - heater_area
percent_heated = (heated_area / area_outside) * 100 if area_outside > 0 else 0

plt.title(f'Область нагрева выше 310K (полезная площадь)\n'
          f'Нагрето {percent_heated:.1f}% площади вне нагревателя')

plt.show()

# ============ ВЫВОД РЕЗУЛЬТАТОВ ============
print(f"\n=== РЕЗУЛЬТАТЫ ===")
print(f"Свойства равномерно перемешанной смеси:")
print(f"  Плотность: {rho_mix:.2f} кг/м³")
print(f"  Теплоёмкость: {cp_mix:.2f} Дж/(кг·К)")
print(f"  Теплопроводность: {lambda_mix:.4f} Вт/(м·К)")
print(f"\nПараметры нагревателя (медь):")
print(f"  Размер: {width_y} x {width_z} м")
print(f"  Мощность: {Q_heater/1e6:.0f} МВт/м³")
print(f"\nРезультаты в конце процесса (t={total_time} с):")
print(f"  Максимальная температура: {np.max(T_final):.1f} K ({np.max(T_final)-273.1:.0f}°C)")
print(f"  Минимальная температура: {np.min(T_final):.1f} K ({np.min(T_final)-273.1:.0f}°C)")
print(f"  Температура в центре нагревателя: {T_final[Ny//2, Nz//2]:.1f} K")
print(f"\nПолезная площадь нагрева (>310K):")
print(f"  Площадь нагревателя: {heater_area:.4f} м²")
print(f"  Площадь вне нагревателя: {area_outside:.4f} м²")
print(f"  Площадь, нагретая выше 310K: {heated_area:.4f} м²")
print(f"  Процент нагретой площади: {percent_heated:.1f}%")