#!/usr/bin/python3.6
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse
import discretizationUtils as grid
import config as config
import operators as op

# --- Setup --- #
config.ny = 6
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
# ---  end  --- #

# Initial conditions
startT = np.expand_dims(np.linspace(0, 1, ny), 1).repeat(nx, axis=1)
startT = np.expand_dims(grid.fieldToVec(startT), 1)
startPsi = np.expand_dims(np.zeros(ny), 1).repeat(nx, axis=1)
startPsi = np.expand_dims(grid.fieldToVec(startPsi), 1)

dyOp = op.dyOp(nx,ny)
dxOp = op.dxOp(nx,ny)
dlOp = op.dlOpStream(nx,ny)

# print(dyOp)
plt.imshow(dlOp.todense())
plt.colorbar()
plt.show()
