import numpy as np
from scipy.optimize import fsolve


def equations(p):
    x = p - np.arange(0, 10)
    return x


x0 = np.arange(0, 10) + 3
y0 = np.arange(0, 10) + 3

xsol = fsolve(equations, x0)

print(equations((xsol)))
