import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import HW1

c_d = 0.25
m = 68.1
g = 9.81
i = 0
delta_t = [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0]
RSME = []
sum = 0
for del_t in delta_t:
    vec_t = np.arange(0, 12 + del_t, del_t)
    for j in range(len(vec_t)):
        sum += (HW1.exact_velocity(c_d, m, vec_t, g)[j] - HW1.forward_Euler_velocity(c_d, m, vec_t, g)[j]) ** 2
    RSME.append(np.sqrt(sum / len(vec_t)))

x = np.array(delta_t)
y = np.array(RSME)

slope, intercept = np.polyfit(x, y, 1)
regression_line = slope * x + intercept

plt.scatter(x, y, label='Data Points')
plt.plot(x, regression_line, color='red', label='Regression Line')

plt.xlabel(r'$\Delta t$')
plt.ylabel('RSME')
plt.title('Linear Regression')
plt.grid()
plt.savefig('p5_regression.png')
plt.show()
x_reshaped = x.reshape(-1, 1)
model = LinearRegression()
model.fit(x_reshaped, y)

y_pred = model.predict(x_reshaped)
r_squared = r2_score(y, y_pred)
print(f'R-squared: {r_squared}')