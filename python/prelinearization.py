# Curve Fit Functions
import scipy
import numpy as np
import matplotlib.pyplot as plt

def exponential(x, a, b):
    return a * np.exp(b * x)

def reciprocal_exponential(x, a, b):
    return a * np.exp(b / x)

x = np.array([i+273 for i in range(25, 75, 5)])
y = np.array([3.97E+02, 2.96E+02, 2.45E+02, 2.04E+02, 1.71E+02, 1.33E+02, 1.09E+02, 9.35E+01, 7.80E+01, 5.41E+01])

plt.figure()
plt.plot(x, y, 'o')

popt, pcov = scipy.optimize.curve_fit(exponential, x, y, p0=[9E7, -0.01])
print(popt)
print(pcov)
# R^2 Value
residuals = y - exponential(x, *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y - np.mean(y))**2)
r_squared = 1 - (ss_res / ss_tot)
print(r_squared)

plt.plot(x, exponential(x, *popt), 'r--', label='exponential fit')

popt, pcov = scipy.optimize.curve_fit(reciprocal_exponential, x, y)
print(popt)
print(pcov)
# R^2 Value
residuals = y - reciprocal_exponential(x, *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y - np.mean(y))**2)
r_squared = 1 - (ss_res / ss_tot)
print(r_squared)

plt.plot(x, reciprocal_exponential(x, *popt), 'g--', label='reciprocal exponential fit')
plt.legend(loc=1)
plt.grid()
plt.xlabel('Temperature (K)')
plt.ylabel('Equilibrium Constant')
plt.title('Equilibrium Constant vs. Temperature')
plt.show()
