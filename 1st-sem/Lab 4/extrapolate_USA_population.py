import statistics
import matplotlib
import math
import numpy  as np
import pandas as pd
from numpy.polynomial.polynomial import Polynomial as Poly

####################################################################################

# GENERAL FUNCTIONS

class LinearEqSystem:
    def __init__(self, matrix, f):
        assert np.shape(matrix)[0] == np.shape(matrix)[1]
        assert np.shape(matrix)[0] == np.shape(f)[0]

        self.matrix    = matrix
        self.f         = f

        self.dimension = np.shape(f)[0]
        self.solution  = np.zeros(self.dimension)
        self.is_solved = False

    def solve_eq(self, solution_method):
        self.solution = solution_method(self)
        self.is_solved = True

def get_b_Newton_coeffs(x, f):
    assert len(x) == len(f), 'len(x) != len(f)'
    b_coeffs = [[0] * (len(f) - i) for i in range(len(f))] # pyramid

    for i, _ in enumerate(f):
        if i == 0:
            for j in range(len(f)):
                b_coeffs[0][j] = f[j]
        else:
            for j in range(len(b_coeffs[i])):
                b_i = b_coeffs[i - 1]
                b_coeffs[i][j] = (b_i[j + 1] - b_i[j]) / (x[j + i] - x[j])

    return b_coeffs

def print_b_Newton_coeffs(b_coeffs):
    df = pd.DataFrame(data=b_coeffs, index=[f'b{i}' for i in range(len(b_coeffs))])
    df = df.transpose()
    df['x'] = years
    cols = list(df.columns)
    cols = cols[len(b_coeffs):] + cols[:len(b_coeffs)]
    df = df.reindex(columns=cols)

    print(df)

def apply_Newton_interpolation(x, x_i, polynomial_coeffs):
    res = 0
    for i, b_i in enumerate(polynomial_coeffs):
        mul = b_i
        for j in range(i):
            mul *= x - x_i[len(x_i) - j - 1]
        res += mul
    return res

def get_spline_linear_system(x, f):
    matrix_size = len(x) - 2
    matrix = np.zeros(shape=(matrix_size, matrix_size))

    for i in range(matrix_size):
        if i != matrix_size - 1:
            matrix[i][i + 1] = x[i + 2] - x[i + 1]

        matrix[i][i] = 2 * (x[i + 2] - x[i])

        if i != 0:
            matrix[i][i - 1] = x[i + 1] - x[i]

    b_col = np.zeros(shape=(matrix_size, 1))
    for i, elem in enumerate(b_col):
        h_i = x[i + 1] - x[i]
        h_i1 = x[i + 2] - x[i + 1]
        b_col[i] = 6 * ((f[i + 2] - f[i + 1]) / h_i1 - (f[i + 1] - f[i]) / h_i)
    
    return LinearEqSystem(np.matrix(matrix), b_col)

def tridiagonal_run_method(linear_system):
    matrix = linear_system.matrix
    f = linear_system.f

    size = linear_system.dimension
    x = np.zeros(size)
    alpha = np.zeros(size)
    beta = np.zeros(size)

    c = matrix[0, 0]
    alpha[0] = - matrix[0, 1] / c
    beta[0] = f[0] / c

    # direct  run through
    for k in range(1, size):
        c = matrix[k, k] + matrix[k, k - 1] * alpha[k - 1]
        if k != size - 1:
            alpha[k] = - matrix[k, k + 1] / c
        beta[k] = (f[k] - matrix[k, k - 1] * beta[k - 1]) / c

    x[-1] = beta[-1]

    for k in range(size - 2, -1, -1):
        x[k] = alpha[k] * x[k + 1] + beta[k]
    
    return x

def get_spline_polynomials(x, f):
    spline_lin_sys_c_k = get_spline_linear_system(x, f)
    spline_lin_sys_c_k.solve_eq(tridiagonal_run_method)

    c_k_sol = spline_lin_sys_c_k.solution

    coeffs = np.zeros(shape=(spline_lin_sys_c_k.dimension + 1, 4))
    for k, coef in enumerate(coeffs):
        h_k = years[k + 1] - years[k]
        coef[0] = f[k + 1]
        c_k = coef[2] = c_k_sol[k] / 2 if k < len(c_k_sol) else 0
        c_k_prev = coeffs[k - 1][2] if k != 0 else 0

        coeffs[k][3] = (c_k - c_k_prev) / (3 * h_k)
        coeffs[k][1] = (f[k + 1] - f[k]) / h_k + (2 * c_k + c_k_prev) * h_k / 3

    return [Poly(coeff) for coeff in coeffs]

####################################################################################

# LAB SPECIFIC STUFF

population_USA = \
{
    1910: 92_228_496,
    1920: 106_021_537,
    1930: 123_202_624,
    1940: 132_164_569,
    1950: 151_325_798,
    1960: 179_323_175,
    1970: 203_211_926,
    1980: 226_545_805,
    1990: 248_709_873,
    2000: 281_421_906,
    2010: 308_745_538,
    2015: 324_607_776
}

years       = list(population_USA.keys())
populations = list(population_USA.values())

population_USA_2022 = 337_168_104

####################################################################################

# SCRIPT START

print('1) Newton interpolation')
print('    Matrix of Newton polynomial coefficients:\n')

b_coeffs = get_b_Newton_coeffs(years, populations)
print_b_Newton_coeffs(b_coeffs)

polynomial_coeffs = [b_i[-1] for b_i in b_coeffs]
population_USA_2022_predicted = int(apply_Newton_interpolation(2022, years, polynomial_coeffs))

print(f'    Predicted population = {population_USA_2022_predicted}')
print(f'    Real population = {population_USA_2022}')
print(f'    Delta = {abs(population_USA_2022 - population_USA_2022_predicted)} ({100 * abs(population_USA_2022 - population_USA_2022_predicted) / population_USA_2022} %)')

print('\n2) Spline interpolation')

spline_polynomials = get_spline_polynomials(years, populations)
population_USA_2022_predicted = int(spline_polynomials[-1](2022 - years[-1]))

print(f'    Predicted population = {population_USA_2022_predicted}')
print(f'    Real population = {population_USA_2022}')
print(f'    Delta = {abs(population_USA_2022 - population_USA_2022_predicted)} ({100 * abs(population_USA_2022 - population_USA_2022_predicted) / population_USA_2022} %)')

# SCRIPT END
