#!/usr/bin/python3.6
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import inv
import discretizationUtils as grid
import config as config
import operators as op
import matrices
import matplotlib.animation as animation

# --- Setup; using the config as a 'global variable module' --- #
config.dt = 0.0001
config.ny = 50
config.nx = 100
config.dy = 1 / (config.ny - 1)
config.dx = 1 / (config.ny - 1)

dt = config.dt
ny = config.ny
nx = config.nx
dy = config.dy
dx = config.dx

# Physical regime
sqrtRa = 7
Tbottom = 1
Ttop = 0
tempDirichlet = [Tbottom, Ttop]
tempNeumann = [0, 0]
# ---  end  --- #

# Initial conditions
startT = np.expand_dims(np.linspace(Tbottom, Ttop, ny), 1).repeat(nx, axis=1)
startT[1:4, 9:11] = 1.1
startT[1:4, 29:31] = 1.1
startT = grid.fieldToVec(startT)
startPsi = np.expand_dims(np.zeros(ny), 1).repeat(nx, axis=1)  # start with zeros for streamfunctions
startPsi = grid.fieldToVec(startPsi)

# Generate operators for differentials (generating them in functions is more computationally expensive)
dyOpPsi = op.dyOp()
dxOpPsi = op.dxOpStream()
dlOpPsi = op.dlOpStreamMod()

dlOpTemp, rhsDlOpTemp = op.dlOpTemp(bcDirArray=tempDirichlet, bcNeuArray=tempNeumann)
dxOpTemp, rhsDxOpTemp = op.dxOpTemp(bcNeuArray=tempNeumann)
dyOpTemp, rhsDyOpTemp = op.dyOpTemp(bcDirArray=tempDirichlet)

Tnplus1 = startT
Psinplus1 = startPsi

toKeep = np.ones((nx * ny,))
toKeep[0:ny] = 0
toKeep[-ny:] = 0
toKeep[ny::ny] = 0
toKeep[ny - 1::ny] = 0
toKeep = sparse.csr_matrix(sparse.diags(toKeep, 0))

fig = plt.figure(figsize=(10, 5), dpi=400)

Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Lars Gebraad'), bitrate=5000)
ims = []

invDlOpPsi = sparse.linalg.inv(dlOpPsi)
print("Done!")

plt.imshow(invDlOpPsi.todense())
plt.show()

# Integrate in time
for it in range(500):
    Tn = Tnplus1
    Psin = Psinplus1
    C, rhsC = matrices.constructC(dxOpTemp=dxOpTemp, rhsDxOpTemp=rhsDxOpTemp, dyOpTemp=dyOpTemp,
                                  rhsDyOpTemp=rhsDyOpTemp,
                                  dlOpTemp=dlOpTemp, rhsDlOpTemp=rhsDlOpTemp, psi=Psin, dxOpPsi=dxOpPsi,
                                  dyOpPsi=dyOpPsi, sqrtRa=sqrtRa)
    # Tnplus1 = inv(sparse.eye(nx*ny) + (dt/2) * C) @ (dt * rhsC + (sparse.eye(nx*ny) - (dt/2) * C) @ Tn)
    Tnplus1 = sparse.linalg.spsolve(sparse.eye(nx * ny) + (dt / 2) * C,
                                    dt * rhsC + (sparse.eye(nx * ny) - (dt / 2) * C) @ Tn, use_umfpack=True)

    # Enforcing column vector
    Tnplus1.shape = (nx * ny, 1)

    # Enforcing Dirichlet
    Tnplus1[0::ny] = Tbottom * 2
    Tnplus1[ny - 1::ny] = Ttop

    Psinplus1 = invDlOpPsi @ (- sqrtRa * toKeep @ (dxOpTemp @ Tnplus1 - rhsDxOpTemp))
    # Psinplus1 = sparse.linalg.spsolve(dlOpPsi, - sqrtRa * toKeep @ (dxOpTemp @ Tnplus1 - rhsDxOpTemp),use_umfpack=True)

    # Enforcing column vector
    Psinplus1.shape = (nx * ny, 1)
    # Reset Dirichlet
    Psinplus1[0::ny] = 0
    Psinplus1[ny - 1::ny] = 0
    Psinplus1[0:ny] = 0
    Psinplus1[-ny:] = 0

    # if (it % 100 == 0):
    #     im = plt.imshow(np.flipud(grid.vecToField(Tnplus1)), animated=True,vmin=-1, vmax=2, aspect=1, cmap=plt.get_cmap('magma'))
    #     ims.append([im])

# plt.imshow(grid.vecToField(dyOpTemp @ startT - rhsDyOpTemp), vmin=-1, vmax=1)
plt.imshow(grid.vecToField(Tnplus1), vmin=0)
plt.colorbar()
plt.show()

# ani = animation.ArtistAnimation(fig, ims, interval=10, blit=True, repeat_delay=0)
# # plt.colorbar()
# ani.save('animation.mp4', writer=writer)
