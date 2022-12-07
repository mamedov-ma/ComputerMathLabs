import matplotlib.pyplot as plt
import math
import numpy as np
import statistics

####################################################################################

# GENERAL FUNCTIONS

class NonlinearEqSystem:
    def __init__(self, get_jacobian, get_sys_equation):

        self.get_jacobian = get_jacobian
        self.get_sys_equation = get_sys_equation
        self.is_solved = False

    def solve_eq(self, initial_vector, stop_iter_epsilon):
        n_iters = 0
        vector_k_next = np.copy(initial_vector)
        vector_k = np.copy(initial_vector) + np.ones(len(initial_vector)) # ||x_k - x_k_next|| > stop_iter_epsilon
        
        while np.abs(np.linalg.norm(vector_k, ord=np.inf) - np.linalg.norm(vector_k_next, ord=np.inf)) > stop_iter_epsilon:
            vector_k = np.copy(vector_k_next)
            vector_k_next = vector_k - np.dot(np.linalg.inv(self.get_jacobian(vector_k)), get_sys_equation(vector_k))
            n_iters += 1
            # print(vector_k, vector_k_next)

        self.is_solved = True

        return vector_k_next, n_iters

def get_jacobian(x_vec):
    return np.array([[-1, -np.sin(x_vec[1])], [np.cos(x_vec[0]), -1]])

def get_sys_equation(x_vec):
    return np.array([np.cos(x_vec[1]) - x_vec[0] + 0.85, np.sin(x_vec[0]) - x_vec[1] - 1.32]).transpose()

####################################################################################

# SCRIPT START

system = NonlinearEqSystem(get_jacobian, get_sys_equation)
roots_vector, n_iters = system.solve_eq(np.array([2, -0.3]), 1e-9)

print('Part 2: iteration Newton method for system of nonlinear equation')

print('  1) Initial system of nonlinear equations:')
print('     sin(x) - y = 1.32')
print('     cos(y) - x = -0.85')
print('  2) Plot of the system show, that there is the single root in area x \in [1.5; 2], y \in [-1; 0]')
print('  3) Iteration process: x_{k+1} = cos(y_k) + 0.85')
print('                        y_{k+1} = sin(x_k) - 1.32')
print('  4) Stop criteria max(abs(x_{k+1} - x_k), abs(y_{k+1} - y_k)) < 1e-9 is used')
print('  5) Calculated roots: x = ' + str(roots_vector[0]) + ', y = ' + str(roots_vector[1]))
print('     Amount of iterations:', n_iters)

# SCRIPT END
