import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ============ СВОЙСТВА МАТЕРИАЛОВ ============
rho_C = 2250; cp_C = 712; lambda_C = 1.59; h_C = 60e6
rho_SiO2 = 1520; cp_SiO2 = 835; lambda_SiO2 = 0.3; h_SiO2 = 14.32e6
rho_H2O = 998.2; cp_H2O = 4182; lambda_H2O = 0.56; h_H2O = 2.4444e6
T_boil = 373.15

materials = {
    'C': {'rho': rho_C, 'cp': cp_C, 'lambda': lambda_C, 'h': h_C, 'name': 'Графит'},
    'SiO2': {'rho': rho_SiO2, 'cp': cp_SiO2, 'lambda': lambda_SiO2, 'h': h_SiO2, 'name': 'Песок'},
    'H2O': {'rho': rho_H2O, 'cp': cp_H2O, 'lambda': lambda_H2O, 'h': h_H2O, 'name': 'Вода'}
}

# ============ ПАРАМЕТРЫ ЗАДАЧИ ============
mass_fractions = {'C': 0.4, 'SiO2': 0.5, 'H2O': 0.1}

# Параметры исходного прямоугольного нагревателя (для расчёта площади)
orig_width_y, orig_width_z = 0.3, 0.4
orig_area = orig_width_y * orig_width_z

# Три круглых нагревателя
num_heaters = 3
single_area = orig_area / num_heaters
heater_radius = np.sqrt(single_area / np.pi)
Q_heater = 1e6

# Параметры расположения
R_heaters = 0.35        # расстояние от центра до центров нагревателей
angles_deg = [0, 120, 240]
angles_rad = np.deg2rad(angles_deg)

# ============ ПАРАМЕТРЫ СЕТКИ (ПОДОБРАНЫ ДЛЯ УСТОЙЧИВОСТИ) ============
total_time = 6000    # с
square_side = 1.0    # м
Ny, Nz = 61, 61      # точек по y и z
Nt = 8000            # шагов по времени

dy = square_side / (Ny - 1)
dz = square_side / (Nz - 1)
dt = total_time / Nt

print("=== ТРИ КРУГЛЫХ НАГРЕВАТЕЛЯ В НЕОДНОРОДНОЙ СРЕДЕ ===")
print(f"Радиус нагревателя: {heater_radius:.4f} м")
print(f"Расстояние от центра: R = {R_heaters} м")
print(f"Углы: {angles_deg}°")
print(f"dy = {dy:.5f} м, dz = {dz:.5f} м")
print(f"dt = {dt:.4f} с, Nt = {Nt}")
print(f"Общее время: {total_time} с")

# Проверка устойчивости
alpha_max = max(lambda_C/(rho_C*cp_C), lambda_SiO2/(rho_SiO2*cp_SiO2), lambda_H2O/(rho_H2O*cp_H2O))
stability = alpha_max * dt * (1/dy**2 + 1/dz**2)
print(f"Число устойчивости: {stability:.4f} (должно быть ≤ 0.5)")

# ============ РАСПРЕДЕЛЕНИЕ КОМПОНЕНТОВ ============
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

# Поля свойств
rho_map = np.zeros((Ny, Nz))
cp_map = np.zeros((Ny, Nz))
lambda_map = np.zeros((Ny, Nz))
water_mass_fraction = np.zeros((Ny, Nz))
has_water = np.zeros((Ny, Nz), dtype=bool)

for i, comp in enumerate(order):
    rho_map[component_masks[i]] = materials[comp]['rho']
    cp_map[component_masks[i]] = materials[comp]['cp']
    lambda_map[component_masks[i]] = materials[comp]['lambda']
    if comp == 'H2O':
        water_mass_fraction[component_masks[i]] = mass_fractions['H2O']
        has_water[component_masks[i]] = True

epsilon = 1e-10
rho_map[rho_map == 0] = materials[order[0]]['rho']
cp_map[cp_map == 0] = materials[order[0]]['cp']
lambda_map[lambda_map == 0] = materials[order[0]]['lambda']

# ============ ТРИ КРУГЛЫХ НАГРЕВАТЕЛЯ ============
center_y = square_side / 2
center_z = square_side / 2
y_coords = np.linspace(0, square_side, Ny)
z_coords = np.linspace(0, square_side, Nz)
Y, Z = np.meshgrid(y_coords, z_coords, indexing='ij')

heater_mask = np.zeros((Ny, Nz), dtype=bool)
for angle_rad in angles_rad:
    hc_y = center_y + R_heaters * np.sin(angle_rad)
    hc_z = center_z + R_heaters * np.cos(angle_rad)
    circle_mask = (Y - hc_y)**2 + (Z - hc_z)**2 <= heater_radius**2
    heater_mask = heater_mask | circle_mask

# ============ ПАРАМЕТРЫ РАСЧЕТА ============
source_val = dt * Q_heater / (rho_map * cp_map + epsilon)
coeff_y_map = lambda_map * dt / ((rho_map * cp_map + epsilon) * dy**2)
coeff_z_map = lambda_map * dt / ((rho_map * cp_map + epsilon) * dz**2)

evaporation_energy = np.zeros((Ny, Nz))
water_evaporated = np.zeros((Ny, Nz), dtype=bool)

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
            delta_T = diffusion + q_source

            if has_water[iy, iz] and not water_evaporated[iy, iz]:
                if T[iy, iz, j] >= T_boil:
                    energy_for_evap = delta_T * cp_map[iy, iz]
                    water_mass = water_mass_fraction[iy, iz] * rho_map[iy, iz]
                    energy_needed = h_H2O * water_mass
                    evaporation_energy[iy, iz] += energy_for_evap * (rho_map[iy, iz] * cp_map[iy, iz])

                    if evaporation_energy[iy, iz] >= energy_needed:
                        water_evaporated[iy, iz] = True
                        leftover = evaporation_energy[iy, iz] - energy_needed
                        T[iy, iz, j+1] = T_boil + leftover / (rho_map[iy, iz] * cp_map[iy, iz])
                        evaporation_energy[iy, iz] = 0
                    else:
                        T[iy, iz, j+1] = T_boil
                else:
                    T[iy, iz, j+1] = T[iy, iz, j] + delta_T
            else:
                T[iy, iz, j+1] = T[iy, iz, j] + delta_T

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

    if (j+1) % 1000 == 0:
        water_count = np.sum(~water_evaporated & has_water)
        print(f"Шаг {j+1}/{Nt} | T_max = {np.max(T_current):.1f}K | Вода: {water_count} | Нагрето: {percent_now:.1f}%")

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

im1 = plt.imshow(comp_display.T, origin='lower', extent=[0, square_side, 0, square_side],
                 aspect='equal', cmap='tab10', vmin=0.5, vmax=len(order)+0.5)
cbar = plt.colorbar(im1, ticks=range(1, len(order)+1), label='Компонент')
cbar.set_ticklabels(comp_labels)
plt.xlabel('y, м')
plt.ylabel('z, м')
plt.title(f'Распределение компонентов (mix={mix_factor})')
plt.show()

# График 2: Расположение нагревателей
plt.figure(figsize=(8, 8))
heaters_display = np.zeros((Ny, Nz))
heaters_display[heater_mask] = 1
plt.imshow(heaters_display.T, origin='lower', extent=[0, square_side, 0, square_side],
           aspect='equal', cmap='Reds', alpha=0.7)
plt.colorbar(label='Нагреватель')
plt.xlabel('y, м')
plt.ylabel('z, м')
plt.title(f'Три круглых нагревателя (R={R_heaters} м, r={heater_radius:.3f} м)')
plt.show()

# График 3: Температурная карта и зоны нагрева (с процентом)
plt.figure(figsize=(14, 6))

T_final = T[:, :, -1]
T_min = np.percentile(T_final, 1)
T_max = np.percentile(T_final, 99)

# Левый график - температурная карта
plt.subplot(1, 2, 1)
im_temp = plt.imshow(T_final.T, origin='lower', extent=[0, square_side, 0, square_side],
                     aspect='equal', cmap='hot', vmin=T_min, vmax=T_max)
plt.colorbar(im_temp, label='Температура, К')
plt.xlabel('y, м')
plt.ylabel('z, м')
plt.title(f'Температура в конце (t={total_time:.0f} с)')

# Правый график - зоны нагрева
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

# Добавляем процент полезной площади на правый график
final_percent = percent_heated_over_time[-1] if percent_heated_over_time else 0
plt.title(f'Зоны нагрева выше 310K\nПолезная площадь нагрева: {final_percent:.1f}%')

plt.tight_layout()
plt.show()

# График 4: Изменение температуры во времени
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

plt.plot(time, temp_center, linewidth=2, label='Центр стержня')
plt.plot(time, temp_heater, linewidth=2, label='Внутри нагревателя')
plt.plot(time, temp_edge, linewidth=2, label='Край стержня (угол)')
plt.axhline(y=310, color='r', linestyle='--', linewidth=1, label='Порог 310 K')
plt.axhline(y=T_boil, color='b', linestyle='--', linewidth=1, label='Кипение воды')
plt.xlabel('Время t, с')
plt.ylabel('Температура T, К')
plt.title(f'Три круглых нагревателя (mix={mix_factor}) | Полезная площадь: {final_percent:.1f}%')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

# График 5: Динамика процента полезной площади
plt.figure(figsize=(12, 6))
time_points = time[1:]
plt.plot(time_points, percent_heated_over_time, linewidth=2, color='green')
plt.xlabel('Время t, с')
plt.ylabel('Площадь нагрева выше 310K, %')
plt.title(f'Динамика полезной площади нагрева (три круглых нагревателя)\nИтог: {final_percent:.1f}%')
plt.grid(True)
plt.ylim(bottom=0)
plt.show()

# ============ ПОДСЧЕТ ============
heated_area = np.sum(heated_above_310) * dy * dz
heater_area = np.sum(heater_mask) * dy * dz
area_outside = 1.0 - heater_area

print(f"\n=== РЕЗУЛЬТАТЫ ===")
print(f"Суммарная площадь нагревателей: {heater_area:.4f} м²")
print(f"Площадь, нагретая выше 310K: {heated_area:.4f} м²")
print(f"Процент нагретой площади: {final_percent:.1f}%")

water_initial = np.sum(has_water)
water_final = np.sum(~water_evaporated & has_water)
print(f"Испарилось воды: {water_initial - water_final} из {water_initial} ячеек")