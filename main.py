#!/usr/bin/python3.6
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg
import discretizationUtils as grid
import config as config
import operators as op
import matrices
import matplotlib.animation as animation
import time

# --- Setup; using the config as a 'global variable module' --- #
config.dt = 0.001
config.nt = 1500
config.ny = 20
config.nx = 30
config.dy = 1 / (config.ny - 1)
config.dx = 1 / (config.ny - 1)

dt = config.dt
nt = config.nt
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

# Takes a long time!
generateAnimation = True
generateEvery = 20
# Speeds up!
useSuperLUFactorizationEq1 = True
useSuperLUFactorizationEq2 = True

# ---  No modifications below this point! --- #

# Initial conditions
startT = np.expand_dims(np.linspace(Tbottom, Ttop, ny), 1).repeat(nx, axis=1)
# startT[1:4, 9:11] = 1.1
# startT[1:4, 29:31] = 1.1
startT = grid.fieldToVec(startT)
startPsi = np.expand_dims(np.zeros(ny), 1).repeat(nx, axis=1)  # start with zeros for streamfunctions
startPsi = grid.fieldToVec(startPsi)

# Generate operators for differentials
dyOpPsi = op.dyOp()
dxOpPsi = op.dxOpStream()
dlOpPsi = op.dlOpStreamMod()
dlOpTemp, rhsDlOpTemp = op.dlOpTemp(bcDirArray=tempDirichlet, bcNeuArray=tempNeumann)
dxOpTemp, rhsDxOpTemp = op.dxOpTemp(bcNeuArray=tempNeumann)
dyOpTemp, rhsDyOpTemp = op.dyOpTemp(bcDirArray=tempDirichlet)

# This matrix is needed to only use the Non-Dirichlet rows in the del²psi operator. The other rows are basically
# ensuring psi_i,j = 0. What it does is it removes rows from a sparse matrix (unit matrix with some missing elements).
psiNonDirichlet = np.ones((nx * ny,))
psiNonDirichlet[0:ny] = 0
psiNonDirichlet[-ny:] = 0
psiNonDirichlet[ny::ny] = 0
psiNonDirichlet[ny - 1::ny] = 0
psiNonDirichlet = sparse.csc_matrix(sparse.diags(psiNonDirichlet, 0))

# Pretty straightforwad; set up the animation stuff
if (generateAnimation):
    # Set up for animation
    fig = plt.figure(figsize=(8.5, 5), dpi=300)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30, metadata=dict(artist='Lars Gebraad'), bitrate=5000)
    ims = []
else:
    fig = plt.figure(figsize=(8.5, 5), dpi=600)

# -- Preconditioning -- #
if (useSuperLUFactorizationEq1):
    start = time.time()
    factor1 = sparse.linalg.factorized(dlOpPsi)  # Makes LU decomposition.
    end = time.time()
    print("Runtime of LU factorization for eq-1: ", end - start, "seconds")

# Set up initial
Tnplus1 = startT
Psinplus1 = startPsi

start = time.time()
# Integrate in time
print("Starting time marching for", nt, "steps ...")
t = [0]
for it in range(nt):
    # if(it == 20):
        # dt = dt*10
    t.append(t[-1] + dt)
    Tn = Tnplus1
    Psin = Psinplus1

    # Regenerate C
    C, rhsC = matrices.constructC(dxOpTemp=dxOpTemp, rhsDxOpTemp=rhsDxOpTemp, dyOpTemp=dyOpTemp,
                                  rhsDyOpTemp=rhsDyOpTemp, dlOpTemp=dlOpTemp, rhsDlOpTemp=rhsDlOpTemp, psi=Psin,
                                  dxOpPsi=dxOpPsi, dyOpPsi=dyOpPsi, sqrtRa=sqrtRa)

    # Solve for Tn+1
    if (useSuperLUFactorizationEq2):
        factor2 = sparse.linalg.factorized(
            sparse.csc_matrix(sparse.eye(nx * ny) + (dt / 2) * C))  # Makes LU decomposition.
        Tnplus1 = factor2(dt * rhsC + (sparse.eye(nx * ny) - (dt / 2) * C) @ Tn)
    else:
        Tnplus1 = sparse.linalg.spsolve(sparse.eye(nx * ny) + (dt / 2) * C,
                                        dt * rhsC + (sparse.eye(nx * ny) - (dt / 2) * C) @ Tn)

    # Enforcing column vector
    Tnplus1.shape = (nx * ny, 1)

    # Enforcing Dirichlet
    Tnplus1[0::ny] = Tbottom
    Tnplus1[ny - 1::ny] = Ttop

    if (useSuperLUFactorizationEq1):
        Psinplus1 = factor1(- sqrtRa * psiNonDirichlet @ (dxOpTemp @ Tnplus1 - rhsDxOpTemp))
    else:
        Psinplus1 = sparse.linalg.spsolve(dlOpPsi, - sqrtRa * psiNonDirichlet @ (dxOpTemp @ Tnplus1 - rhsDxOpTemp))

    # Enforcing column vector
    Psinplus1.shape = (nx * ny, 1)
    # Reset Dirichlet
    Psinplus1[0::ny] = 0
    Psinplus1[ny - 1::ny] = 0
    Psinplus1[0:ny] = 0
    Psinplus1[-ny:] = 0

    if (it % generateEvery == 0):
        if (generateAnimation):
            ux = grid.vecToField(dyOpPsi @ Psinplus1)
            uy = grid.vecToField(dxOpPsi @ Psinplus1)
            maxSpeed = np.max(np.square(ux) + np.square(uy))
            plt.quiver(np.flipud(ux), -np.flipud(uy))
            im = plt.imshow(np.flipud(grid.vecToField(Tnplus1)), vmin=0, aspect=1, cmap=plt.get_cmap('magma'), vmax=2)
            clrbr = plt.colorbar()
            plt.tight_layout()
            plt.title("t: %.2f, it: %i, Maximum speed: %.2f" % (t[-1], it, maxSpeed))
            plt.savefig("fields/field%i.png" % (it / generateEvery))
            plt.gca().clear()
            clrbr.remove()
            # ims.append([im])
        print("time step:", it, end='\r')

end = time.time()
print("Runtime of time-marching: ", end - start, "seconds")

# And output figure(s)
if (generateAnimation):
    # ani = animation.ArtistAnimation(fig, ims, interval=10, blit=True, repeat_delay=0)
    # plt.colorbar()
    # plt.tight_layout()
    # plt.title("Maximum speed: %.2f" % maxSpeed)
    # ani.save('animation.mp4', writer=writer)
    a = 1
else:
    ux = grid.vecToField(dyOpPsi @ Psinplus1)
    uy = grid.vecToField(dxOpPsi @ Psinplus1)
    maxSpeed = np.max(np.square(ux) + np.square(uy))
    plt.quiver(np.flipud(ux), -np.flipud(uy))
    plt.imshow(np.flipud(grid.vecToField(Tnplus1)), cmap=plt.get_cmap('magma'))
    plt.colorbar()
    plt.tight_layout()
    plt.title("Maximum speed: %.2f" % maxSpeed)
    plt.savefig("Convection-field.png")
