#!/usr/bin/python3.6
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse
import discretizationUtils as grid
import config as config
import operators as op
import matrices

# --- Setup --- #
config.ny = 6
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
# ---  end  --- #

# Initial conditions
startT = np.expand_dims(np.linspace(0, 1, ny), 1).repeat(nx, axis=1)
startT = np.expand_dims(grid.fieldToVec(startT), 1)
startPsi = np.expand_dims(np.zeros(ny), 1).repeat(nx, axis=1)
startPsi = np.expand_dims(grid.fieldToVec(startPsi), 1)

# Generate operators for differentials (generating them in functions is more computationally expensive)
dyOpPsi = op.dyOp(nx, ny)
dxOpPsi = op.dxOpStream(nx, ny)
dlOpPsi = op.dlOpStream(nx, ny)
dlOpTemp, rhsDlOpTemp = op.dlOpTemp(nx, ny, [Tbottom, Ttop], [0, 0])
dxOpTemp, rhsDxOpTemp = op.dxOpTemp(nx, ny, [0, 0])
dyOpTemp, rhsDyOpTemp = op.dyOpTemp(nx, ny, [Tbottom, Ttop])


C, rhsC = matrices.constructC(dxOpTemp=dxOpTemp, rhsDxOpTemp=rhsDxOpTemp, dyOpTemp=dyOpTemp, rhsDyOpTemp=rhsDyOpTemp,
                              psi=startPsi, dxOpPsi=dxOpPsi, dyOpPsi=dyOpPsi, sqrtRa=sqrtRa)

print(rhsC)
