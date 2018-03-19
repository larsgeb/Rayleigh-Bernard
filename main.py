#!/usr/bin/python3.6
import matplotlib.pyplot as plt
import numpy as np

import boundary
import config
import discretizationUtils as grid
import kernels

config.ny = 11
config.nx = config.ny * 2
config.dy = 1
config.dx = 1

ny = config.ny
nx = config.nx
dy = config.dy
dx = config.dx

sqrt_ra = 7

Tbottom = 1
Ttop = 0

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


dt = 0.5
dT = laplaceMatT @ currentT + laplaceVecT - sqrt_ra * (
        np.multiply( (dyMatPsi @ currentPsi + dyVecPsi), (dxMatT @ currentT + dxVecT) )
        -
        np.multiply( (dxMatPsi @ currentPsi + dxVecPsi), (dyMatT @ currentT + dyVecT) )
)

nextT = currentT + dT * dt

currentT[2] = 2

plt.matshow(grid.vecToField(currentT))
plt.colorbar()
plt.show()

plt.matshow(grid.vecToField(dyMatT @ currentT + dyVecT[::-1]))
plt.colorbar()
plt.show()