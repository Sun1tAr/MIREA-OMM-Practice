**ВАРИАНТ 4**

Задание 1


```python
# Задача 1
import numpy as np
from scipy.optimize import minimize

def f1(vars):
    x, y = vars
    return np.log(1 + x**2 + y**2)

start = [0.5, -1.0]
bounds = [(-2, 2), (-2, 2)]

res1 = minimize(f1, start, method='L-BFGS-B', bounds=bounds)

print("Задача 1: f(x,y) = ln(1 + x² + y²)")
print("Ограничения: -2 ≤ x ≤ 2, -2 ≤ y ≤ 2")
print(f"Ответ: x = {res1.x[0]:.6f}, y = {res1.x[1]:.6f}, f_min = {res1.fun:.6f}")
```

    Задача 1: f(x,y) = ln(1 + x² + y²)
    Ограничения: -2 ≤ x ≤ 2, -2 ≤ y ≤ 2
    Ответ: x = 0.000000, y = 0.000000, f_min = 0.000000
    

Задание 2


```python
# Задача 2
import numpy as np
from scipy.optimize import minimize, LinearConstraint

def f2(vars):
    x, y = vars
    return (x-3)**2 + (y+2)**2

x0 = [1.0, 2.0]
bnds = [(0, 5), (0, 5)]
constr = LinearConstraint([[2, 1]], ub=6)

res2 = minimize(f2, x0, method='SLSQP', bounds=bnds, constraints=constr)

print("Задача 2: f(x,y) = (x-3)² + (y+2)²")
print("Ограничения: 0 ≤ x ≤ 5, 0 ≤ y ≤ 5, 2x + y ≤ 6")
print(f"Ответ: x = {res2.x[0]:.4f}, y = {res2.x[1]:.4f}, f_min = {res2.fun:.4f}")
```

    Задача 2: f(x,y) = (x-3)² + (y+2)²
    Ограничения: 0 ≤ x ≤ 5, 0 ≤ y ≤ 5, 2x + y ≤ 6
    Ответ: x = 3.0000, y = 0.0000, f_min = 4.0000
    

Задание 3


```python
# Задача 3
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

costs = [30000, 20000, 50000, 10000, 40000, 25000]
profits = [40000, 25000, 60000, 12000, 50000, 30000]

res3 = milp(c=-np.array(profits),
            constraints=LinearConstraint([costs], ub=[100000]),
            integrality=np.ones(6),
            bounds=Bounds(0, 1))

selected = res3.x.astype(bool)
total_p = np.array(profits)[selected].sum()
total_c = np.array(costs)[selected].sum()

print("Задача 3: Максимизация прибыли от проектов")
print("Ограничения: бюджет ≤ 100000, бинарный выбор")
print(f"Ответ: проекты {np.where(selected)[0]}, прибыль = {total_p}, затраты = {total_c}")
```

    Задача 3: Максимизация прибыли от проектов
    Ограничения: бюджет ≤ 100000, бинарный выбор
    Ответ: проекты [0 1 3 4], прибыль = 127000, затраты = 100000
    
