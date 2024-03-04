import numpy as np
import math

# Problem1
def exact_velocity(c_d, m, vec_t, g):
    #vec_t is an array
    v = np.empty(len(vec_t))
    for i in range(len(vec_t)):
        v[i] = float(np.sqrt(g * m / c_d) * np.tanh(np.sqrt(g * c_d / m) * vec_t[i]))
    
    return v # v is an array

# Problem3
def forward_Euler_velocity(c_d, m, vec_t, g):
    #numerical method of v' = g - c/m * v^2
    i = 1
    ln = len(vec_t)
    v = np.empty(ln)
    while i < ln:
        delta_t = vec_t[i] - vec_t[i - 1]
        v[i] = v[i - 1] + delta_t * (g - c_d / m * (v[i - 1] ** 2))
        i += 1
    #print(len(v) == len(vec_t))
    return v


# Part 1
# matrix multiplication
def mat_mat_mul(a, b): #a and b are matrices
    # using np.array([]) lets make a matrix
    # make a list inside the array
    # a * b = M
    if len(a[0]) != len(b):
        return print('Dimension Mismatch')
    else:
        m, n = len(a), len(b[0])
        M = [[0 for _ in range(n)] for _ in range(m)]
        for i in range(len(a)):
            for j in range(len(b[0])):
                for k in range(len(a[i])):
                    M[i][j] += a[i][k] * b[k][j]
    
        return np.array(M)

# Part2
# Approximating sin(x) by Maclaurin Series
def approximate_sin(x, es, maxit):
    # Initialize variables
    approx_sin = 0
    old_approx_sin = 0
    n = 0
    
    # Iterate until max iterations or desired accuracy
    while n < maxit:
        # Compute the nth term of the Maclaurin series
        term = ((-1) ** n) * (x ** (2 * n + 1)) / math.factorial(2 * n + 1)
        
        # Update the approximation
        approx_sin += term
        
        # Calculate relative error
        if old_approx_sin != 0:
            relative_error = abs((approx_sin - old_approx_sin) / approx_sin)
            if relative_error <= es:
                break
        
        # Update old approximation
        old_approx_sin = approx_sin
        
        # Move to the next term
        n += 1
    
    return approx_sin, relative_error, n



