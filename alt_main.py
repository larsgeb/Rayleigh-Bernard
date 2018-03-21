#!/usr/bin/python3.6
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse
import scipy.linalg as sparselinalg
from scipy.sparse.linalg import inv
import discretizationUtils as grid
import config as config
import operators as op
import matrices

dt = 0.05

# --- Setup --- #
config.ny = 80
config.nx = config.ny * 2
config.dy = 1
config.dx = 1
# config.dy = 1/(config.ny-1)
# config.dx = 1/(config.ny-1)
ny = config.ny
nx = config.nx
dy = config.dy
dx = config.dx

sqrtRa = 7
Tbottom = 1
Ttop = 0
tempDirichlet = [Tbottom, Ttop]
# ---  end  --- #

# Initial conditions
# start with linear profile for temperature
startT = np.expand_dims(np.linspace(Tbottom, Ttop, ny), 1).repeat(nx, axis=1)
startT = grid.fieldToVec(startT)
startPsi = np.expand_dims(np.zeros(ny), 1).repeat(nx, axis=1)  # start with zeros for streamfunctions
startPsi = grid.fieldToVec(startPsi)

# Generate operators for differentials (generating them in functions is more computationally expensive)
dyOpPsi = op.dyOp()
dxOpPsi = op.dxOpStream()
dlOpPsi = op.dlOpStreamMod()

dlOpTemp, rhsDlOpTemp = op.dlOpTemp(bcDirArray=tempDirichlet, bcNeuArray=[0, 0])
dxOpTemp, rhsDxOpTemp = op.dxOpTemp(bcNeuArray=[0, 0])
dyOpTemp, rhsDyOpTemp = op.dyOpTemp(bcDirArray=tempDirichlet)

Tnplus1 = startT
Psinplus1 = startPsi

toKeep = np.ones((nx * ny,))
toKeep[0:ny] = 0
toKeep[-ny:] = 0
toKeep[ny::ny] = 0
toKeep[ny-1::ny] = 0
toKeep = sparse.csr_matrix(sparse.diags(toKeep, 0))

# Integrate in time
for it in range(100):
    Tn = Tnplus1
    Psin = Psinplus1
    C, rhsC = matrices.constructC(dxOpTemp=dxOpTemp, rhsDxOpTemp=rhsDxOpTemp, dyOpTemp=dyOpTemp, rhsDyOpTemp=rhsDyOpTemp,
                                  dlOpTemp=dlOpTemp, rhsDlOpTemp=rhsDlOpTemp, psi=Psin, dxOpPsi=dxOpPsi,
                                  dyOpPsi=dyOpPsi, sqrtRa=sqrtRa)
    # Tnplus1 = inv(sparse.eye(nx*ny) + (dt/2) * C) @ (dt * rhsC + (sparse.eye(nx*ny) - (dt/2) * C) @ Tn)
    Tnplus1 = sparse.linalg.spsolve(sparse.eye(nx * ny) + (dt / 2) * C,
                                    dt * rhsC + (sparse.eye(nx * ny) - (dt / 2) * C) @ Tn)
    Tnplus1.shape = (nx*ny,1)
    Psinplus1 = sparse.linalg.spsolve(dlOpPsi,-sqrtRa * toKeep @ (dxOpTemp @ Tnplus1 - rhsDxOpTemp))
    Psinplus1.shape = (nx*ny,1)

plt.imshow(grid.vecToField(Tnplus1))
plt.show()
