import matplotlib
import math
import numpy as np
import statistics

####################################################################################

# GENERAL FUNCTIONS

def iterate_nonlinear_eq(iter_process, x):
    return iter_process(x)

def solve_nonlinear_eq(iter_process, x_0, stop_iter_epsilon):
    n_iters = 0
    x_k = x_0
    x_k_next = x_0 + 2 * stop_iter_epsilon # abs(x_k - x_k_next) > stop_iter_epsilon

    while np.abs(x_k_next - x_k) > stop_iter_epsilon:
        x_k = x_k_next
        x_k_next = iter_process(x_k)
        n_iters += 1

    return x_k_next, n_iters

####################################################################################

# LAB SPECIFIC FUNCTIONS

# Equation: arctan(x - 1) + 2x = 0
def iteration_process(x):
    return -0.5 * np.arctan(x - 1)

####################################################################################

# SCRIPT START

root, n_iters = solve_nonlinear_eq(iteration_process, 1, 1e-16)

print('Part 1: iteration method for nonlinear equation')

print('  1) Initial nonlinear equation: arctan(x - 1) + 2x = 0 => arctan(x - 1) = -2x => 1 root')
print('  2) Localization area: L = [0; 1]')
print('  3) Iteration process: x = -0.5 * arctan(x - 1) => x_{k+1} = -0.5 * arctan(x_k - 1)')
print('     Derivative analysis: (-0.5 * arctan(x - 1))\' = -0.5/((x - 1)^2 + 1) => |-0.5/((x - 1)^2 + 1)| < 0.5 < 1 => is possible')
print('  4) Stop criteria abs(x_{k+1} - x_k) < 1e-16 is used')
print('  5) Calculated root: x =', root)
print('     Amount of iterations:', n_iters)

# SCRIPT END
