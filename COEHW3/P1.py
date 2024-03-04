import numpy as np
import matplotlib.pyplot as plt

x = np.array([0.1, 0.2, 0.4, 0.6, 0.9, 1.3, 1.5, 1.7, 1.8])
y = np.array([0.75, 1.25, 1.45, 1.25, 0.85, 0.55, 0.35, 0.28, 0.18])
x_space = np.linspace(0.1, 1.8, 100)

sum0 = 0 # sum for x^2
for i in range(len(x)):
    sum0 += x[i] ** 2
A = np.array([[9, sum(x)],
              [sum(x), sum0]])

sum1 = 0 # sum for ln(y/x)
for j in range(len(x)):
    sum1 += np.log(y[j] / x[j])

sum2 = 0 # sum for x * ln(y/x)
for k in range(len(x)):
    sum2 += x[k] * np.log(y[k]/x[k])

b = np.array([[sum1],
              [sum2]])
alpha_1 = np.linalg.solve(A, b)[0]
alpha_2 = np.linalg.solve(A, b)[1]
alpha_1 = np.exp(alpha_1)
print(alpha_1, alpha_2)
S_r = 0
for l in range(len(x)):
    S_r += float((y[l] - alpha_1 * x[l] * np.exp(alpha_2 * x[l])) ** 2)
print(S_r)
plt.plot(x, y, 'o', color = 'darkblue',label = 'Real Data')
plt.plot(x_space, alpha_1 * x_space * np.exp(alpha_2 * x_space), color = 'r',
         label = r'Fit function $y = \alpha_1 x e^{\alpha_2 x}$')
plt.title(rf'$S_r$ = {S_r: .5f}')
plt.xlim(0, 2)
plt.grid(True)
plt.legend(fontsize = 14)
plt.savefig('./COEHW3/plot1.png')
plt.show()