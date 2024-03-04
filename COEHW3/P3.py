import numpy as np
import matplotlib.pyplot as plt
from HW3 import least_square_fourier

def series(coe, x, omega_o): # Fourier Series
    sum_ = coe[0]
    for i in range(1, len(coe)):
        if i % 2 == 0:
            sum_ += coe[i] * np.sin(i/2 * omega_o * x)
        elif i % 2 != 0:
            sum_ += coe[i] * np.cos((i+1)/2 * omega_o * x)
    
    return sum_


x = np.linspace(0, 2 * np.pi, 200)
y =  - 7+ 2 * np.cos(x) - 3 * np.sin(x) + np.cos(2*x) + 0.5 * np.sin(2*x) + np.random.normal(0, 0.5, len(x))

a = least_square_fourier(x, y, 2, 1)
print(a)

plt.plot(x, y, '*')
plt.plot(x, series(a, x, 1), color = 'y', label = 'Fitted Fourier Series')
plt.plot(x, - 7+ 2 * np.cos(x) - 3 * np.sin(x) + np.cos(2*x) + 0.5 * np.sin(2*x), 'red', label = 'Real Function')

plt.legend(fontsize = 15)
plt.grid()
plt.show()

