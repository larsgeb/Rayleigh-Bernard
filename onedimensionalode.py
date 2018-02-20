#!/usr/bin/python3.6
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as optimize

import boundary
import config
import discretizationUtils as grid
import kernels

config.ny = 10
config.nx = 10
config.dy = 0.1
config.dx = 0.1

ny = config.ny
nx = config.nx
dy = config.dy
dx = config.dx

## Definition of boundary value problem
neuPsi = None
dirPsi = boundary.Dirichlet(y0=0)

# Define matrices needed for every derivative in systems of algebraic equations using programmed kernels and bc's
dyMatPsi, dyVecPsi = kernels.makeSparsedyKernel(dirichlet=dirPsi, neumann=neuPsi)

kernels.plotKernel(dyMatPsi, dyVecPsi)


# Define residual
def toMinimize(solution):
    return (dyMatPsi @ solution - dyVecPsi - 1)


startT = np.expand_dims(np.zeros(ny), 1).repeat(nx, axis=1)

sol = optimize.broyden2(toMinimize, np.vstack(
    (np.expand_dims(grid.fieldToVec(startT), 1), np.expand_dims(grid.fieldToVec(startT), 1))), f_tol=6e-6)

plt.matshow(grid.vecToField(sol))
plt.colorbar()
plt.show()
