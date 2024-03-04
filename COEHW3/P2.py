import numpy as np
import matplotlib.pyplot as plt
from HW3 import least_square_poly

x = np.array([0.1, 0.2, 0.4, 0.6, 0.9, 1.3, 1.5, 1.7, 1.8])
y = np.array([0.75, 1.25, 1.45, 1.25, 0.85, 0.55, 0.35, 0.28, 0.18])


a = least_square_poly(x, y, 4)
c = np.polyfit(x, y, 4)
a_fit = np.poly1d(a[::-1])
c_fit = np.poly1d(c)
print(a[::-1])
print(c)
x_ = np.linspace(0, 2, 200)

plt.plot(x_, a_fit(x_), color = 'r')
plt.plot(x_, c_fit(x_), color = 'b')
plt.plot(x, y, 'o')

plt.ylim(0, 2)
plt.show()