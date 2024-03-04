import numpy as np
error = 'error'

def least_square_poly(x, y, k):
    if len(x) != len(y):
        return error
    Z = np.zeros((len(x), k+1))
    for i in range(len(x)):
        for j in range(k+1):
            Z[i, j] = x[i] ** j

    a = np.linalg.solve(np.dot(Z.T, Z), np.dot(Z.T, y))
    return a # to use np.poly1d we need to change the order

def least_square_fourier(x, y, k, omega_o):
    if len(x) != len(y):
        return error
    Z = np.zeros((len(x), 2 * k + 1))
    for i in range(len(x)):
        for j in range(2 * k + 1):
            if j == 0:
                Z[i, j] = 1
            elif j % 2 == 0:
                Z[i, j] = np.sin(j / 2 * omega_o * x[i])
            elif j != 0 and j % 2 == 1:
                Z[i, j] = np.cos((j+1) / 2 * omega_o * x[i])
    
    a = np.linalg.solve(np.dot(Z.T, Z), np.dot(Z.T, y))
    return a

def my_dft(x):
    n = len(x)
    e = np.zeros(n)
    for i in range(n):
        e[i] = np.exp(-1j * i * (2 * np.pi / n))
    F = np.dot(x, e)
    return F