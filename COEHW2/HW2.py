import numpy as np
error = 'error'

# decomposition of a matrix
def naive_LU(A):
    if len(A) != len(A[0]):
        return error
    else:
        n = len(A)
        # Initialize L as an identity matrix and U as a zero matrix
        L = np.zeros((n, n))
        U = np.zeros((n, n))

        # Perform LU decomposition
        for i in range(n):
            # Upper triangular matrix
            for k in range(i, n):
                sum_ = sum(L[i, j] * U[j, k] for j in range(i))
                U[i, k] = A[i, k] - sum_

            # Lower triangular matrix
            for k in range(i, n):
                if i == k:
                    L[i, i] = 1.0
                else:
                    sum_ = sum(L[k, j] * U[j, i] for j in range(i))
                    L[k, i] = (A[k, i] - sum_) / U[i, i]

    return L, U

# Let's solve equation
# Problem 7
def solve_LU(L, U, b):
    # first solve Ld = b
    # d = Ux
    if len(L) != len(L[0]) or len(U) != len(U[0]) or len(L) != len(U):
        return error
    n = len(L)
    d = np.zeros(n)
    x = np.zeros(n)
    for i in range(n):
        d[i] = (b[i] - np.dot(L[i, :i], d[:i])) / L[i, i]
    
    for j in range(n-1, -1, -1):
        x[j] = (d[j] - np.dot(U[j, j+1:], x[j+1:])) / U[j, j]
    return x
print
# Problem 11
# Finding inverse of A using LU decomposition
# inverse of A = inv(U)inv(L)
def inv_using_naive_LU(A):
    if np.linalg.det(A) < 1.0e-9:
        return error
    A_inv = np.zeros((len(A), len(A[0])))
    L, U = naive_LU(A)
    for i in range(len(A)):
        I = np.zeros(len(A))
        I[i] = 1
        x = solve_LU(L, U, I)
        for j in range(len(A_inv)):
            A_inv[j, i] = x[j]

    return A_inv

def Richardson_it(A, b, omega, tol, max_it):
    # norm of matrix np.linalg.norm(A, ord = 2)
    I = np.eye(len(A))
    x = np.zeros(len(A)) # first guess
    if len(A) != len(A[0]):
        return error
    
    elif np.linalg.norm(I - omega * A, ord = 2) >= 1:
        return error
    else:
        N = 0
        error_ = np.linalg.norm(np.dot(A, x) - b, ord = 2)
        while N <= max_it and error_ > tol:
            new_x = x + omega * (b - np.dot(A, x))
            x = new_x
            error = np.linalg.norm(np.dot(A, x) - b, ord = 2)
            N += 1
            
    
    return x, N, error_

# Problem 15
def largest_eig(A, tol, maxit):
    if len(A) != len(A[0]):
        return error
    x = np.array([1, 1, 1]) # initial guess
    i = 0
    lambda_ = [1] # lets just set the first eigenvalue as 1(initial eigenvalue = 1)
    error_ = 1
    while i < maxit and error_ > tol:
        x = np.dot(A, x) / max(abs(np.dot(A, x)))
        
        lambda_.append(max(abs(np.dot(A, x))))
        
        error_ = abs((lambda_[i+1] - lambda_[i])/lambda_[i+1])
        i += 1
    
    return lambda_[len(lambda_) - 1], x, i, error_

def my_Cholesky(A): # Docomposition to U_T * U
    if len(A) != len(A[0]):
        return error

    n = len(A)
    U = np.zeros((n, n))
    
    for i in range(len(A)):
        for j in range(len(A)):
            if A[i, j] != A[j, i]:
                return error
    for i in range(n):
        for j in range(i+1):
            if i == j:
                U[i][j] = np.sqrt(A[i][i] - sum(U[i][k] ** 2 for k in range(j)))
            else:
                U[i][j] = (A[i][j] - sum(U[i][k] * U[j][k] for k in range(j))) / U[j][j]
    
    return np.transpose(U)

def my_GaussSiedel(A, b, tol, max_it):
    if len(A) != len(b):
        return error
    n = len(b)
    x = np.zeros(n)
    x_new = np.zeros(n)
    error_ = tol + 1  # Initialize error to be greater than epsilon
    iteration = 0

    while error_ > tol and iteration < max_it: # let's try using np.allclose
        for i in range(n):
            sum1 = np.dot(A[i, :i], x_new[:i])
            sum2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - sum1 - sum2) / A[i, i]

        error_ = np.linalg.norm(np.dot(A, x_new) - b, ord = 2)
        x = x_new
        iteration += 1

    return x

