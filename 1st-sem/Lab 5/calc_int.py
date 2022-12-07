import statistics
import matplotlib
import math
import numpy  as np


def calculate_by_trapeze(step, func_vals):
    assert step >= 0
    return (sum(func_vals) - (func_vals[0] + func_vals[-1]) / 2) * step

def calculate_by_Runge(step, func_vals):
    assert step >= 0
    integral_trapeze = calculate_by_trapeze(step, func_vals)

    sparse_func_vals = \
        np.array([func_vals[2 * i] for i in range(len(func_vals) // 2 + len(func_vals) % 2)])
    integral_trapeze_sparse = calculate_by_trapeze(2 * step, sparse_func_vals)

    return integral_trapeze + (integral_trapeze - integral_trapeze_sparse) / (2 ** 2 - 1)

def calculate_by_Simpson(step, func_vals):
    assert step >= 0
    odd_sum  = 0
    even_sum = 0

    for i in range(1, len(func_vals) - 1):
        if i % 2 == 0:
            even_sum += func_vals[i]
        else:
            odd_sum += func_vals[i]

    return (func_vals[0] + 4 * odd_sum + 2 * even_sum + func_vals[-1]) * step / 3


step = 0.25
func_vals = np.array([1.0, 0.989616, 0.958851, 0.908852, 0.841471, 0.759188, 
                    0.664997, 0.562278, 0.454649])


print('Simpson formula: ', calculate_by_Simpson(step, func_vals))
print('Trapeze formula: ', calculate_by_trapeze(step, func_vals))
print('Runge formula (rule): ', calculate_by_Runge(step, func_vals))


