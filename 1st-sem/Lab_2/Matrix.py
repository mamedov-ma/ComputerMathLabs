import numpy as np             
import matplotlib.pyplot as plt

n = 20
diag = 10

class Norm:

    #возвращает максимальную сумму в строке
    def max_sum(matrix):
        max = np.sum(np.abs(matrix[0]))
        for line in matrix:
            cur_sum = np.sum(np.abs(line))
            if cur_sum > max:
                max = cur_sum
        return max

    def __get_first_norm(matrix):
        matr_size = matrix.shape
        if len(matr_size) != 1:
            return Norm.max_sum(matrix)
        else:
            return np.amax(np.abs(matrix))

    def get_norm(matrix):
        return Norm.__get_first_norm(matrix)


class UpperRelax:

    def __init__(self, A, f):
        self.A = A
        self.f = f
        self.teta = 1.5
        self.epsilon = 0.01
        matr_size = A.shape
        if (len(matr_size) != 2) or (matr_size[0] != matr_size[1]):
            print('Incorrect matrix')
        self.n = matr_size[0]
        self.r_arr = np.array([])

    def __get_initial_x(self):
        x = np.copy(self.f)
        return x

    #подсчет невязки
    def __calc_r(self, x):
        r = Norm.get_norm(np.matmul(self.A, x) - self.f)
        self.r_arr = np.append(self.r_arr, r)

    #концы включены
    #х - массив
    def __sum(self, start, end, x, j):
        sum = 0.0
        for k in range(end - start + 1):
            sum += self.A[j, start + k] / self.A[j, j] * x[start + k]
        return sum

    #индексация с 0
    def __iteration(self, x_prev):
        x_new = np.array([])
        for j in range(self.n):
            x_j = (-1) * self.teta * self.__sum(0, j - 1, x_new, j) + (1 - self.teta) * x_prev[j] \
                - self.teta * self.__sum(j + 1, self.n - 1, x_prev, j) + self.teta * self.f[j] / self.A[j, j]
            x_new = np.append(x_new, x_j)
        return x_new
    
    def calculate(self):
        num_of_iter = 0.0
        x_prev = self.__get_initial_x()
        x_new = np.array([])
        prev_error = np.Inf
        curr_error = 0.0
        while True:
            x_new = self.__iteration(x_prev)
            self.__calc_r(x_new)
            curr_error = Norm.get_norm(x_new - x_prev)
            if curr_error < self.epsilon:
                print('конечная разность:', Norm.get_norm(x_new - x_prev))
                break
            else:
                x_prev = x_new
            if prev_error < curr_error:
                print('Warning: on iteration', num_of_iter, 'error was', prev_error, ' and now', curr_error)
            prev_error = curr_error
            num_of_iter += 1
        return x_new


#метод Холецкого
class SqrtMethod:

    def __init__(self, A, f):
        self.A = A
        self.f = f
        self.L = np.full((n, n), 0.0)
        self.L_T = np.full((n, n), 0.0)
        matr_size = A.shape
        if (len(matr_size) != 2) or (matr_size[0] != matr_size[1]):
            print('Incorrect matrix')
        self.n = matr_size[0]
        self.r_arr = np.array([])

    #концы включены
    def __sum(self, end, left_index_1, left_index_2):
        sum = 0.0
        for k in range(end + 1):
            sum += self.L[left_index_1, k] * self.L[left_index_2, k]
        return sum

    def __get_L_elem(self, i, j):
        if i == j:
            return np.sqrt(self.A[i, i] - self.__sum(i - 1, i, i))
        else:
            return (self.A[i, j] - self.__sum(j - 1, j, i)) / self.L[j, j]

    def __fill_L(self):
        for i in range(n):
            for j in range(i + 1):
                self.L[i, j] = self.__get_L_elem(i, j)
        self.L_T = np.copy(self.L)
        self.L_T = np.transpose(self.L_T)
    
    #ytne +1 поскольку при вызове добавляется 1 тк в массиве нумерация с 0
    def __sum_y(self, num_of_iter, cur_index, y):
        sum = 0.0
        for k in range(num_of_iter):
            sum += self.L[cur_index, k] * y[k] 
        return sum

    #решение уравнения Ly=f (обратный ход) 
    #L - нижне-треугольная
    def find_y(self):
        y = np.array([])
        for i in range(self.n):
            y_i = (self.f[i] - self.__sum_y(i - 1 + 1, i, y)) / self.L[i, i]   
            y = np.append(y, y_i)
        return y

    #тут не надо + 1 (количество итераций достаточно)
    def __sum_x(self, num_of_iter, cur_index, x):
        sum = 0.0
        for p in range(num_of_iter):
            sum += self.L_T[cur_index, cur_index + p + 1] * x[cur_index + p + 1]
        return sum

    #решение уравнения L^T x = y
    def find_x(self, y):
        x = np.full(self.n, 0.0)
        for k in range(self.n):
            x[self.n - k - 1] = (y[n - k - 1] - self.__sum_x(k, self.n - k - 1, x)) \
                / self.L_T[n - k - 1, n - k - 1]
        return x

    def calculate(self):
        self.__fill_L()
        y = self.find_y()
        x = self.find_x(y)
        return x

#скалярное произведение с Г=А xAy 
def scalar_by_A(x, A, y):
    first = np.matmul(x, A)
    return np.matmul(first, y)

#подсчет минимального и максимального собственного значения
#на википедии неправильный метод
def calc_lambda(A):
    num_of_iter = 200
    y_new = np.copy(A[0])
    y_prev = np.copy(A[0])
    for i in range(num_of_iter):
        y_prev = y_new
        mult = np.matmul(A, y_prev) 
        y_new = mult #/ Norm.get_norm(mult) #так было на вики
    return  Norm.get_norm(y_new) / Norm.get_norm(y_prev)  #scalar_by_A(np.transpose(y_new), A, y_new) / scalar_by_A(np.transpose(y_prev), A, y_prev) 

#если перемножить получится единичная матрица с точностью до 10^-18
def determine_koef(A):
    rev_A = np.linalg.inv(A)
    return Norm.get_norm(A) * Norm.get_norm(rev_A)

def get_line(line_num):
    line = np.array([])
    for j in range(n):
        if line_num == j:
            line = np.append(line, diag)
        else:    
            line = np.append(line, 1 / (line_num + j + 2))
    return line

def generate_matrix():
    matr = get_line(0)
    for i in range(n - 1):
        line = get_line(i + 1)
        matr = np.vstack((matr, line))
    return matr

def generate_rhs():
    rhs = np.array([])
    for i in range(n):
        rhs = np.append(rhs, 1 / (i + 1))
    return np.transpose(rhs)

def iteration_method(A, f):
    method = UpperRelax(A, f)
    method.calculate()


def test(A, vec):
    print('Init vec', vec)
    num_of_iter = 200
    y_new = np.copy(vec)
    y_prev = np.copy(vec)
    for i in range(num_of_iter):
        y_prev = y_new
        mult = np.matmul(A, y_prev) 
        y_new = mult #/ Norm.get_norm(mult) #так было на вики
    return  Norm.get_norm(y_new) / Norm.get_norm(y_prev)

def main():
    A = generate_matrix()
    f = generate_rhs()
    method = UpperRelax(A, f)
    ans_iter = method.calculate()
    print('Ответ методом верхних релаксаций:', ans_iter)
    straight_method = SqrtMethod(A, f)
    ans_straight = straight_method.calculate()
    print('Ответ методом Холецкого:', ans_straight)
    print('Невязка:', method.r_arr)
    print('')
    print('Число обусловленности', determine_koef(A))
    print('Максиммальное собственное значение:', calc_lambda(A))
    print('Минимальное собственное значение через A^-1:', 1 / calc_lambda(np.linalg.inv(A)))
    #print('Минимальное собственное значение через A^T:', calc_lambda(np.transpose(A)))
    num, vec =  np.linalg.eigh(A)
    print('')
    print('Реальные собственные значения для :', num)
    print('Лямбда через другой вектор:', test(A, vec[0]))

if __name__ == '__main__':
    main()