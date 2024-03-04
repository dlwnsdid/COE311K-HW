import numpy as np
import matplotlib.pyplot as plt
import HW1

c_d = 0.25
m = 68.1
vec_t = np.arange(0, 12.5, 0.5)
g = 9.81
plt.plot(vec_t, HW1.exact_velocity(c_d, m, vec_t, g), color = "Red", label = "Exact")
plt.plot(vec_t, HW1.forward_Euler_velocity(c_d, m, vec_t, g), color = "Blue", label = "Numerical")
plt.title(r'Exact Veclocity $\sqrt{\frac{gm}{c_d}}tanh(\sqrt{\frac{gc_d}{m}}t)$ vs. Numerical Method')
plt.xlabel(r'Time $(s)$')
plt.ylabel(r'Velocity $(v)$')
plt.legend()
plt.grid()
plt.savefig('p4.png')
plt.show()
