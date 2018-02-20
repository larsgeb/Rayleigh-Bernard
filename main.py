#!/usr/bin/python3.6
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.sparse.linalg as linalgSP
from scipy.sparse import hstack, vstack

import boundary
import config
import discretizationUtils as grid
import kernels

config.ny = 5
config.nx = config.ny * 2
config.dy = 1
config.dx = 1

ny = config.ny
nx = config.nx
dy = config.dy
dx = config.dx

sqrt_ra = 7

Tbottom = 50
Ttop = 25

# Initial conditions
startT = np.expand_dims(np.linspace(0, 1, ny), 1).repeat(nx, axis=1)
startT = np.expand_dims(grid.fieldToVec(startT), 1)

startPsi = np.expand_dims(np.zeros(ny), 1).repeat(nx, axis=1)
startPsi = np.expand_dims(grid.fieldToVec(startPsi), 1)

## Definition of boundary value problem
dirTemp = boundary.Dirichlet(y0=Tbottom, yend=Ttop)
neuTemp = boundary.Neumann(xend=0, x0=0)
dirPsi = boundary.Dirichlet(x0=0, y0=0, xend=0, yend=0)

# Define matrices needed for every derivative in systems of algebraic equations using programmed kernels and bc's
laplaceMatPsi, laplaceVecPsi = kernels.makeSparseLaplaceKernel(dirichlet=dirPsi, neumann=None)
laplaceMatT, laplaceVecT = kernels.makeSparseLaplaceKernel(dirichlet=dirTemp, neumann=neuTemp)

dxMatPsi, dxVecPsi = kernels.makeSparsedxKernel(dirichlet=dirPsi, neumann=None)
dxMatT, dxVecT = kernels.makeSparsedxKernel(dirichlet=dirTemp, neumann=neuTemp)

dyMatPsi, dyVecPsi = kernels.makeSparsedyKernel(dirichlet=dirPsi, neumann=None)
dyMatT, dyVecT = kernels.makeSparsedyKernel(dirichlet=dirTemp, neumann=neuTemp)


# Define residual
def toMinimize(input):
    global sqrt_ra
    global solution_n_min_1

    psi_p, t_p = np.split(solution_n_min_1, 2)
    psi, t = np.split(input, 2)

    a = laplaceMatPsi @ psi + sqrt_ra * dxMatT @ t - (laplaceVecPsi + sqrt_ra * dxVecT)
    b = sqrt_ra * (
            np.multiply(dyMatPsi @ psi, dxMatT @ t) - np.multiply(dxMatPsi @ psi, dyMatT @ t)
    ) - laplaceMatPsi @ psi - (sqrt_ra * (dxVecT.multiply(dyVecPsi) - dyVecT.multiply(dxVecPsi)) + laplaceVecPsi)

    # a_prev = laplaceMatPsi @ psi_p + sqrt_ra * dxMatT @ t_p - (laplaceVecPsi + sqrt_ra * dxVecT)

    b_prev = sqrt_ra * (
            np.multiply(dyMatPsi @ psi_p, dxMatT @ t_p) - np.multiply(dxMatPsi @ psi_p, dyMatT @ t_p)
    ) - laplaceMatPsi @ psi_p - (sqrt_ra * (dxVecT.multiply(dyVecPsi) - dyVecT.multiply(dxVecPsi)) + laplaceVecPsi)

    return np.concatenate((a, b - b_prev))


currentPsi = startPsi
currentT = startT

currentSol = np.concatenate((currentPsi, currentT))

# Once
A = laplaceMatPsi
B = sqrt_ra * dxMatT
topRHS = laplaceVecPsi + dxVecT

# Redo
psi_x = (dxMatPsi @ currentPsi).repeat(nx * ny, axis=1)
psi_y = (dyMatPsi @ currentPsi).repeat(nx * ny, axis=1)

C = laplaceMatT + sqrt_ra * (np.multiply(psi_y, dxMatT.todense().A) - np.multiply(psi_x, dyMatT.todense().A))
botRHS = laplaceVecT + sqrt_ra * (dxVecT - dyVecT)

top = hstack((A, B))
bot = hstack((scipy.sparse.dok_matrix((nx * ny, nx * ny), dtype='float'), C))

tot = vstack((top, bot))

inverse = linalgSP.inv(scipy.sparse.csc_matrix(tot))

plt.matshow(tot.todense().A)
plt.colorbar()
plt.show()

plt.matshow(inverse.todense().A)
plt.colorbar()
plt.show()
