import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ============ СВОЙСТВА МАТЕРИАЛОВ ============
# Графит (C)
rho_C = 2250      # кг/м³
cp_C = 712        # Дж/(кг·К)
lambda_C = 1.59   # Вт/(м·К)
h_C = 60e6        # Дж/кг (теплота реакции/сгорания)

# Сухой песок (SiO2)
rho_SiO2 = 1520   # кг/м³
cp_SiO2 = 835     # Дж/(кг·К)
lambda_SiO2 = 0.3 # Вт/(м·К)
h_SiO2 = 14.32e6  # Дж/кг

# Вода (H2O)
rho_H2O = 998.2   # кг/м³
cp_H2O = 4182     # Дж/(кг·К)
lambda_H2O = 0.56 # Вт/(м·К)
h_H2O = 2.4444e6  # Дж/кг (скрытая теплота парообразования!)
T_boil = 373.15   # K (температура кипения воды)

# Словари свойств
materials = {
    'C': {'rho': rho_C, 'cp': cp_C, 'lambda': lambda_C, 'h': h_C, 'name': 'Графит'},
    'SiO2': {'rho': rho_SiO2, 'cp': cp_SiO2, 'lambda': lambda_SiO2, 'h': h_SiO2, 'name': 'Песок'},
    'H2O': {'rho': rho_H2O, 'cp': cp_H2O, 'lambda': lambda_H2O, 'h': h_H2O, 'name': 'Вода'}
}

# ============ ПАРАМЕТРЫ ЗАДАЧИ ============
# Массовые доли (из файла: C=40%, SiO2=50%, H2O=10%)
mass_fractions = {
    'C': 0.4,
    'SiO2': 0.5,
    'H2O': 0.1
}

# Параметры нагревателя (исходный прямоугольник)
orig_width_y = 0.3   # по y, м
orig_width_z = 0.4   # по z, м
Q_heater = 1e6       # мощность нагревателя, Вт/м³

# Параметры сетки
square_side = 1.0   # сторона квадратного сечения, м
total_time = 6000    # время процесса, с

# Сетка
Ny = 51
Nz = 51
Nt = 2000

dy = square_side / (Ny - 1)
dz = square_side / (Nz - 1)
dt = total_time / Nt

# ============ УПРАВЛЕНИЕ РАСПРЕДЕЛЕНИЕМ КОМПОНЕНТОВ ============
order = ['C', 'SiO2', 'H2O']  # Графит внутри, песок в середине, вода снаружи
mix_factor = 1  # 0 - слои, 1 - равномерно

# ============ ФУНКЦИЯ РАСПРЕДЕЛЕНИЯ КОМПОНЕНТОВ ============
def get_component_masks(Ny, Nz, order, mix_factor):
    y_coords = np.linspace(0, 1, Ny)
    Y = np.ones((Nz, 1)) @ y_coords.reshape(1, -1)
    Y = Y.T

    # Слои
    layer_masks = []
    layer_width = 1.0 / len(order)
    for i in range(len(order)):
        y_start = i * layer_width
        y_end = (i + 1) * layer_width
        layer_mask = (Y >= y_start) & (Y < y_end)
        layer_masks.append(layer_mask)

    # Равномерное распределение
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

    # Смешивание
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

    # Нормализация
    combined = np.zeros((Ny, Nz), dtype=int)
    for i, mask in enumerate(final_masks):
        combined[mask] = i + 1
    combined[combined == 0] = 1

    return [combined == (i + 1) for i in range(len(order))]

# ============ ИНИЦИАЛИЗАЦИЯ ПОЛЕЙ ============
component_masks = get_component_masks(Ny, Nz, order, mix_factor)

# Поля свойств
rho_map = np.zeros((Ny, Nz))
cp_map = np.zeros((Ny, Nz))
lambda_map = np.zeros((Ny, Nz))

# ДОПОЛНИТЕЛЬНО: массовая доля воды и флаг наличия воды
water_mass_fraction = np.zeros((Ny, Nz))  # массовая доля воды в ячейке
has_water = np.zeros((Ny, Nz), dtype=bool)

for i, comp in enumerate(order):
    rho_map[component_masks[i]] = materials[comp]['rho']
    cp_map[component_masks[i]] = materials[comp]['cp']
    lambda_map[component_masks[i]] = materials[comp]['lambda']
    if comp == 'H2O':
        water_mass_fraction[component_masks[i]] = mass_fractions['H2O']
        has_water[component_masks[i]] = True

# Защита от нулей
epsilon = 1e-10
rho_map[rho_map == 0] = materials[order[0]]['rho']
cp_map[cp_map == 0] = materials[order[0]]['cp']
lambda_map[lambda_map == 0] = materials[order[0]]['lambda']

# ============ НАГРЕВАТЕЛЬ ============
center_y = square_side / 2
center_z = square_side / 2
y_coords = np.linspace(0, square_side, Ny)
z_coords = np.linspace(0, square_side, Nz)
Y, Z = np.meshgrid(y_coords, z_coords, indexing='ij')

heater_mask = (Y >= center_y - orig_width_y/2) & (Y <= center_y + orig_width_y/2) & \
              (Z >= center_z - orig_width_z/2) & (Z <= center_z + orig_width_z/2)

# ============ ПАРАМЕТРЫ РАСЧЕТА ============
source_val = dt * Q_heater / (rho_map * cp_map + epsilon)
coeff_y_map = lambda_map * dt / ((rho_map * cp_map + epsilon) * dy**2)
coeff_z_map = lambda_map * dt / ((rho_map * cp_map + epsilon) * dz**2)

# ДОПОЛНИТЕЛЬНЫЕ ПОЛЯ ДЛЯ ФАЗОВОГО ПЕРЕХОДА
# Накопленная энергия на испарение воды в каждой ячейке [Дж/кг]
evaporation_energy = np.zeros((Ny, Nz))
# Флаг: вся вода уже испарилась
water_evaporated = np.zeros((Ny, Nz), dtype=bool)

# Массив температур
T = np.zeros((Ny, Nz, Nt))
T[:, :, 0] = 293.0

# Массив для хранения процента полезной площади по времени
percent_heated_over_time = []

# ============ РАСЧЕТ ============
print("\n=== НЕОДНОРОДНАЯ СРЕДА С ФАЗОВЫМ ПЕРЕХОДОМ ВОДЫ ===")
print(f"Порядок компонентов: {order}")
print(f"Коэффициент смешения: {mix_factor}")
print(f"Температура кипения воды: {T_boil - 273.15:.1f}°C")
print(f"Размер сетки: {Ny}x{Nz}")
print(f"dt = {dt:.3f} с")
print("\nНачало расчета...")

for j in range(0, Nt-1):
    for iy in range(1, Ny-1):
        for iz in range(1, Nz-1):
            # Тепло от нагревателя
            q_source = source_val[iy, iz] if heater_mask[iy, iz] else 0

            # Теплопроводность
            diffusion = coeff_y_map[iy, iz] * (T[iy+1, iz, j] - 2*T[iy, iz, j] + T[iy-1, iz, j]) + \
                        coeff_z_map[iy, iz] * (T[iy, iz+1, j] - 2*T[iy, iz, j] + T[iy, iz-1, j])

            # Предварительное изменение температуры (без учета фазового перехода)
            delta_T = diffusion + q_source

            # УЧЕТ ФАЗОВОГО ПЕРЕХОДА ВОДЫ
            if has_water[iy, iz] and not water_evaporated[iy, iz]:
                # Вода еще есть
                if T[iy, iz, j] >= T_boil:
                    # Достигли температуры кипения - энергия идет на испарение
                    # Энергия, доступная для испарения [К]
                    energy_for_evap = delta_T * cp_map[iy, iz]

                    # Эквивалентная масса воды в ячейке [кг/м³]
                    water_mass = water_mass_fraction[iy, iz] * rho_map[iy, iz]

                    # Энергия, необходимая для испарения всей воды [Дж/кг * кг/м³ = Дж/м³]
                    energy_needed = h_H2O * water_mass

                    # Накопленная энергия на испарение
                    evaporation_energy[iy, iz] += energy_for_evap * (rho_map[iy, iz] * cp_map[iy, iz])

                    if evaporation_energy[iy, iz] >= energy_needed:
                        # Вся вода испарилась
                        water_evaporated[iy, iz] = True
                        # Остаток энергии идет на нагрев
                        leftover = evaporation_energy[iy, iz] - energy_needed
                        T[iy, iz, j+1] = T_boil + leftover / (rho_map[iy, iz] * cp_map[iy, iz])
                        evaporation_energy[iy, iz] = 0
                    else:
                        # Вода еще испаряется, температура не растет
                        T[iy, iz, j+1] = T_boil
                else:
                    # Ниже температуры кипения - обычный нагрев
                    T[iy, iz, j+1] = T[iy, iz, j] + delta_T
            else:
                # Воды нет или она уже испарилась - обычный нагрев
                T[iy, iz, j+1] = T[iy, iz, j] + delta_T

    # Граничные условия: теплоизоляция
    T[0, :, j+1] = T[1, :, j+1]
    T[-1, :, j+1] = T[-2, :, j+1]
    T[:, 0, j+1] = T[:, 1, j+1]
    T[:, -1, j+1] = T[:, -2, j+1]

    # Подсчет процента полезной площади нагрева в данный момент
    T_current = T[:, :, j+1]
    heated_now = (T_current > 310) & (~heater_mask)
    heated_area_now = np.sum(heated_now) * dy * dz
    total_area_outside = square_side**2 - np.sum(heater_mask) * dy * dz
    percent_now = (heated_area_now / total_area_outside) * 100 if total_area_outside > 0 else 0
    percent_heated_over_time.append(percent_now)

    if (j+1) % 500 == 0:
        water_count = np.sum(~water_evaporated & has_water)
        print(f"Шаг {j+1}/{Nt} | T_max = {np.max(T_current):.1f}K | Ячеек с водой: {water_count} | Нагрето: {percent_now:.1f}%")

print("Расчет завершен!")

# ============ ГРАФИКИ ============
y = np.linspace(0, square_side, Ny)
z = np.linspace(0, square_side, Nz)
time = np.linspace(0, total_time, Nt)

# ГРАФИК 1: Распределение компонентов и теплопроводности
plt.figure(figsize=(12, 4))

comp_display = np.zeros((Ny, Nz))
comp_labels = []
for i, comp in enumerate(order):
    comp_display[component_masks[i]] = i + 1
    comp_labels.append(materials[comp]['name'])

plt.subplot(1, 2, 1)
im1 = plt.imshow(comp_display.T, origin='lower', extent=[0, square_side, 0, square_side],
                 aspect='equal', cmap='tab10', vmin=0.5, vmax=len(order)+0.5)
cbar = plt.colorbar(im1, ticks=range(1, len(order)+1), label='Компонент')
cbar.set_ticklabels(comp_labels)
plt.xlabel('y, м')
plt.ylabel('z, м')
plt.title(f'Распределение компонентов (mix_factor={mix_factor})')

plt.subplot(1, 2, 2)
im2 = plt.imshow(lambda_map.T, origin='lower', extent=[0, square_side, 0, square_side],
                 aspect='equal', cmap='hot')
plt.colorbar(im2, label='λ, Вт/(м·К)')
plt.xlabel('y, м')
plt.ylabel('z, м')
plt.title('Теплопроводность λ(y,z)')

plt.tight_layout()
plt.show()

# ГРАФИК 2: Сечение температуры в конце процесса
plt.figure(figsize=(12, 4))

T_final = T[:, :, -1]
T_min = np.percentile(T_final, 1)
T_max = np.percentile(T_final, 99)

plt.subplot(1, 2, 1)
im3 = plt.imshow(T_final.T, origin='lower', extent=[0, square_side, 0, square_side],
                 aspect='equal', cmap='hot', vmin=T_min, vmax=T_max)
plt.colorbar(im3, label='Температура, К')
plt.xlabel('y, м')
plt.ylabel('z, м')
plt.title(f'Температура в конце (t={total_time:.0f} с)')

# ГРАФИК 3: Зоны нагрева выше 310K
heated_above_310 = (T_final > 310) & (~heater_mask)
display_mask = np.zeros((Ny, Nz))
display_mask[heated_above_310] = 1
display_mask[heater_mask] = 2

plt.subplot(1, 2, 2)
cmap_custom = ListedColormap(['darkblue', 'yellow', 'red'])
plt.imshow(display_mask.T, origin='lower', extent=[0, square_side, 0, square_side],
           aspect='equal', cmap=cmap_custom, norm=plt.Normalize(0, 2))
plt.colorbar(ticks=[0.25, 1, 2], label='Зона',
             format=plt.FuncFormatter(lambda x, _: {0.25: 'Холодная (<310K)',
                                                    1: 'Нагретая (>310K)',
                                                    2: 'Нагреватель'}[x]))
plt.xlabel('y, м')
plt.ylabel('z, м')
plt.title('Зоны нагрева выше 310K')

plt.tight_layout()
plt.show()

# ГРАФИК 4: Изменение температуры во времени
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
plt.axhline(y=T_boil, color='b', linestyle='--', linewidth=1, label='Температура кипения воды')
plt.xlabel('Время t, с')
plt.ylabel('Температура T, К')
plt.title(f'Неоднородная среда с фазовым переходом (mix_factor={mix_factor})')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

# ГРАФИК 5: Процент полезной площади нагрева по времени
plt.figure(figsize=(12, 6))
time_points = time[1:]  # пропускаем t=0
plt.plot(time_points, percent_heated_over_time, linewidth=2, color='green')
plt.xlabel('Время t, с')
plt.ylabel('Площадь нагрева выше 310K, %')
plt.title(f'Динамика полезной площади нагрева (mix_factor={mix_factor})')
plt.grid(True)
plt.ylim(bottom=0)
plt.show()

# ============ ПОДСЧЕТ ПОЛЕЗНОЙ ПЛОЩАДИ ============
heated_area = np.sum(heated_above_310) * dy * dz
heater_area = np.sum(heater_mask) * dy * dz
total_area = square_side * square_side
area_outside_heater = total_area - heater_area
percent_heated_final = (heated_area / area_outside_heater) * 100 if area_outside_heater > 0 else 0

print(f"\n=== ПОДСЧЕТ ПОЛЕЗНОЙ ПЛОЩАДИ НАГРЕВА ===")
print(f"Пороговая температура: 310 K")
print(f"Температура кипения воды: {T_boil - 273.15:.1f}°C")
print(f"Коэффициент смешения: {mix_factor}")
print(f"Порядок компонентов: {order}")
print(f"\nОбщая площадь сечения: {total_area:.4f} м²")
print(f"Площадь нагревателя: {heater_area:.4f} м²")
print(f"Площадь вне нагревателя: {area_outside_heater:.4f} м²")
print(f"Площадь, нагретая выше 310K: {heated_area:.4f} м²")
print(f"Процент нагретой площади: {percent_heated_final:.1f}%")

# Информация об испарении воды
water_cells_initial = np.sum(has_water)
water_cells_final = np.sum(~water_evaporated & has_water)
print(f"\n=== ИСПАРЕНИЕ ВОДЫ ===")
print(f"Ячеек с водой в начале: {water_cells_initial}")
print(f"Ячеек с водой в конце: {water_cells_final}")
print(f"Испарилось: {water_cells_initial - water_cells_final} ячеек")