import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ============ СВОЙСТВА МАТЕРИАЛОВ СРЕДЫ ============
rho_C = 2250; cp_C = 712; lambda_C = 1.59
rho_SiO2 = 1520; cp_SiO2 = 835; lambda_SiO2 = 0.3
rho_H2O = 998.2; cp_H2O = 4182; lambda_H2O = 0.56

# ============ СВОЙСТВА НАГРЕВАТЕЛЯ (МЕДЬ) ============
rho_heater = 8900      # кг/м³
cp_heater = 380        # Дж/(кг·К)
lambda_heater = 400    # Вт/(м·К)

materials = {
    'C': {'rho': rho_C, 'cp': cp_C, 'lambda': lambda_C, 'name': 'Графит'},
    'SiO2': {'rho': rho_SiO2, 'cp': cp_SiO2, 'lambda': lambda_SiO2, 'name': 'Песок'},
    'H2O': {'rho': rho_H2O, 'cp': cp_H2O, 'lambda': lambda_H2O, 'name': 'Вода'}
}

# ============ ПАРАМЕТРЫ ЗАДАЧИ ============
mass_fractions = {'C': 0.4, 'SiO2': 0.5, 'H2O': 0.1}

# Параметры прямоугольного нагревателя
width_y = 0.3      # м
width_z = 0.4      # м
Q_heater = 5e7     # 50 МВт/м³

# ============ ПАРАМЕТРЫ СЕТКИ (t=600 с) ============
total_time = 600    # 600 секунд (10 минут)
square_side = 1.0   # м
Ny, Nz = 61, 61
Nt = 4000           # шагов по времени (dt = 0.15 с)

dy = square_side / (Ny - 1)
dz = square_side / (Nz - 1)
dt = total_time / Nt

print("=== ПРЯМОУГОЛЬНЫЙ НАГРЕВАТЕЛЬ (замена материала) ===")
print(f"Q_heater = {Q_heater/1e6:.0f} МВт/м³")
print(f"total_time = {total_time} с")
print(f"dt = {dt:.4f} с, Nt = {Nt}")

# Проверка устойчивости
alpha_heater = lambda_heater / (rho_heater * cp_heater)
stability = alpha_heater * dt * (1/dy**2 + 1/dz**2)
print(f"Число устойчивости для нагревателя: {stability:.4f} (должно быть ≤ 0.5)")

# ============ РАСПРЕДЕЛЕНИЕ КОМПОНЕНТОВ СРЕДЫ ============
order = ['C', 'SiO2', 'H2O']
mix_factor = 1.0

def get_component_masks(Ny, Nz, order, mix_factor):
    y_coords = np.linspace(0, 1, Ny)
    Y = np.ones((Nz, 1)) @ y_coords.reshape(1, -1)
    Y = Y.T

    layer_masks = []
    layer_width = 1.0 / len(order)
    for i in range(len(order)):
        y_start = i * layer_width
        y_end = (i + 1) * layer_width
        layer_mask = (Y >= y_start) & (Y < y_end)
        layer_masks.append(layer_mask)

    np.random.seed(42)
    random_field = np.random.random((Ny, Nz))
    uniform_masks = []
    cum_frac = 0
    for comp in order:
        cum_frac += mass_fractions[comp]
        if comp == order[0]:
            uniform_mask = random_field < cum_frac
        else:
            uniform_mask = (random_field >= (cum_frac - mass_fractions[comp])) & (random_field < cum_frac)
        uniform_masks.append(uniform_mask)

    final_masks = []
    np.random.seed(42)
    for i in range(len(order)):
        layer_mask = layer_masks[i]
        uniform_mask = uniform_masks[i]
        random_for_blend = np.random.random((Ny, Nz))
        use_layer = random_for_blend > mix_factor
        use_uniform = random_for_blend <= mix_factor
        final_mask = (use_layer & layer_mask) | (use_uniform & uniform_mask)
        final_masks.append(final_mask)

    combined = np.zeros((Ny, Nz), dtype=int)
    for i, mask in enumerate(final_masks):
        combined[mask] = i + 1
    combined[combined == 0] = 1

    return [combined == (i + 1) for i in range(len(order))]

component_masks = get_component_masks(Ny, Nz, order, mix_factor)

# ============ ИНИЦИАЛИЗАЦИЯ ПОЛЕЙ ============
rho_map = np.zeros((Ny, Nz))
cp_map = np.zeros((Ny, Nz))
lambda_map = np.zeros((Ny, Nz))

for i, comp in enumerate(order):
    rho_map[component_masks[i]] = materials[comp]['rho']
    cp_map[component_masks[i]] = materials[comp]['cp']
    lambda_map[component_masks[i]] = materials[comp]['lambda']

# ============ ПРЯМОУГОЛЬНЫЙ НАГРЕВАТЕЛЬ ============
center_y = square_side / 2
center_z = square_side / 2
y_coords = np.linspace(0, square_side, Ny)
z_coords = np.linspace(0, square_side, Nz)
Y, Z = np.meshgrid(y_coords, z_coords, indexing='ij')

heater_mask = (Y >= center_y - width_y/2) & (Y <= center_y + width_y/2) & \
              (Z >= center_z - width_z/2) & (Z <= center_z + width_z/2)

# ============ ЗАМЕНА МАТЕРИАЛА В ЗОНЕ НАГРЕВАТЕЛЯ ============
rho_map[heater_mask] = rho_heater
cp_map[heater_mask] = cp_heater
lambda_map[heater_mask] = lambda_heater

# ============ ПАРАМЕТРЫ РАСЧЕТА ============
epsilon = 1e-10
source_val = dt * Q_heater / (rho_map * cp_map + epsilon)
coeff_y_map = lambda_map * dt / ((rho_map * cp_map + epsilon) * dy**2)
coeff_z_map = lambda_map * dt / ((rho_map * cp_map + epsilon) * dz**2)

T = np.zeros((Ny, Nz, Nt))
T[:, :, 0] = 293.0
percent_heated_over_time = []

# ============ РАСЧЕТ ============
print("\nНачало расчета...")

for j in range(0, Nt-1):
    for iy in range(1, Ny-1):
        for iz in range(1, Nz-1):
            q_source = source_val[iy, iz] if heater_mask[iy, iz] else 0
            diffusion = coeff_y_map[iy, iz] * (T[iy+1, iz, j] - 2*T[iy, iz, j] + T[iy-1, iz, j]) + \
                        coeff_z_map[iy, iz] * (T[iy, iz+1, j] - 2*T[iy, iz, j] + T[iy, iz-1, j])
            T[iy, iz, j+1] = T[iy, iz, j] + diffusion + q_source

    T[0, :, j+1] = T[1, :, j+1]
    T[-1, :, j+1] = T[-2, :, j+1]
    T[:, 0, j+1] = T[:, 1, j+1]
    T[:, -1, j+1] = T[:, -2, j+1]

    T_current = T[:, :, j+1]
    heated_now = (T_current > 310) & (~heater_mask)
    heated_area_now = np.sum(heated_now) * dy * dz
    total_area_outside = 1.0 - np.sum(heater_mask) * dy * dz
    percent_now = (heated_area_now / total_area_outside) * 100 if total_area_outside > 0 else 0
    percent_heated_over_time.append(percent_now)

    if (j+1) % 500 == 0:
        print(f"Шаг {j+1}/{Nt} | T_max = {np.max(T_current):.1f}K | Нагрето: {percent_now:.1f}%")

print("Расчет завершен!")

# ============ ГРАФИКИ ============
y = np.linspace(0, square_side, Ny)
z = np.linspace(0, square_side, Nz)
time = np.linspace(0, total_time, Nt)

# График 1: Распределение компонентов
plt.figure(figsize=(10, 8))
comp_display = np.zeros((Ny, Nz))
comp_labels = []
for i, comp in enumerate(order):
    comp_display[component_masks[i]] = i + 1
    comp_labels.append(materials[comp]['name'])
comp_display[heater_mask] = len(order) + 1
comp_labels.append('Нагреватель (медь)')

im1 = plt.imshow(comp_display.T, origin='lower', extent=[0, square_side, 0, square_side],
                 aspect='equal', cmap='tab10', vmin=0.5, vmax=len(order)+1.5)
cbar = plt.colorbar(im1, ticks=range(1, len(order)+2), label='Компонент')
cbar.set_ticklabels(comp_labels)
plt.xlabel('y, м')
plt.ylabel('z, м')
plt.title(f'Распределение компонентов (нагреватель заменяет материал)')
plt.show()

# График 2: Температурная карта и зоны нагрева
plt.figure(figsize=(14, 6))

T_final = T[:, :, -1]
T_min = np.percentile(T_final, 1)
T_max = np.percentile(T_final, 99)

plt.subplot(1, 2, 1)
im_temp = plt.imshow(T_final.T, origin='lower', extent=[0, square_side, 0, square_side],
                     aspect='equal', cmap='hot', vmin=T_min, vmax=T_max)
plt.colorbar(im_temp, label='Температура, К')
plt.xlabel('y, м')
plt.ylabel('z, м')
plt.title(f'Температура в конце (t={total_time:.0f} с)')

plt.subplot(1, 2, 2)
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

final_percent = percent_heated_over_time[-1] if percent_heated_over_time else 0
plt.title(f'Зоны нагрева выше 310K\nПолезная площадь нагрева: {final_percent:.1f}%')

plt.tight_layout()
plt.show()

# График 3: Изменение температуры
plt.figure(figsize=(12, 6))

center_y_idx = Ny // 2
center_z_idx = Nz // 2
temp_center = T[center_y_idx, center_z_idx, :]

heater_points = np.where(heater_mask)
if len(heater_points[0]) > 0:
    temp_heater = T[heater_points[0][0], heater_points[1][0], :]
else:
    temp_heater = np.zeros_like(temp_center)

temp_edge = T[0, 0, :]

plt.plot(time, temp_center, linewidth=2, label='Центр стержня (среда)')
plt.plot(time, temp_heater, linewidth=2, label='Внутри нагревателя')
plt.plot(time, temp_edge, linewidth=2, label='Край стержня (угол)')
plt.axhline(y=310, color='r', linestyle='--', linewidth=1, label='Порог 310 K')
plt.xlabel('Время t, с')
plt.ylabel('Температура T, К')
plt.title(f'Прямоугольный нагреватель | Полезная площадь: {final_percent:.1f}%')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

# График 4: Динамика процента
plt.figure(figsize=(12, 6))
time_points = time[1:]
plt.plot(time_points, percent_heated_over_time, linewidth=2, color='green')
plt.xlabel('Время t, с')
plt.ylabel('Площадь нагрева выше 310K, %')
plt.title(f'Динамика полезной площади нагрева\nИтог: {final_percent:.1f}%')
plt.grid(True)
plt.ylim(bottom=0)
plt.show()

print(f"\n=== РЕЗУЛЬТАТЫ ===")
print(f"Мощность нагревателя: {Q_heater/1e6:.0f} МВт/м³")
print(f"Процент нагретой площади: {final_percent:.1f}%")
print(f"Максимальная температура: {np.max(T_final):.1f} K ({np.max(T_final)-273.1:.0f}°C)")