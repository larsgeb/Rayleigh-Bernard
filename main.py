#!/usr/bin/python3.6
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg
import discretizationUtils as grid
import config as config
import operators as op
import matrices
import time

# --- Setup; using the config as a 'global variable module' --- #
config.dt = 0.001
config.nt = 3000
config.ny = 20
config.nx = 40
config.dy = 1 / (config.ny - 1)
config.dx = 2 / (config.ny - 1)

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

generateAnimation = True  # Takes a long time!
generateEvery = 100
generateFinal = False
useSuperLUFactorizationEq1 = True  # doesn't do much as it needs to be done every loop
useSuperLUFactorizationEq2 = True  # Speeds up!
quiverEveryNPoints = 1
# ---  No modifications below this point! --- #

# Initial conditions
startT = np.expand_dims(np.linspace(Tbottom, Ttop, ny), 1).repeat(nx, axis=1)
startT[2, int(nx / 2 - 1):int(nx / 2)] = 0.5
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
    gs = gridspec.GridSpec(3, 2)
    gs.update(wspace=0.5, hspace=0.75)
    ax1 = plt.subplot(gs[0:2, :], )
    ax2 = plt.subplot(gs[2, 0])
    ax3 = plt.subplot(gs[2, 1])
else:
    fig = plt.figure(figsize=(8.5, 5), dpi=600)

# -- Preconditioning -- #
if (useSuperLUFactorizationEq1):
    start = time.time()
    # Makes LU decomposition.
    factor1 = sparse.linalg.factorized(dlOpPsi)
    end = time.time()
    print("Runtime of LU factorization for eq-1: %.2e seconds" % (end - start))

# Set up initial
Tnplus1 = startT
Psinplus1 = startPsi

# initialize accumulators
t = []
maxSpeedArr = []
heatFluxConductionArr = []
heatFluxAdvectionArr = []
totalHeatFluxArr = []

# Integrate in time
start = time.time()
print("Starting time marching for", nt, "steps ...")
for it in range(nt):
    if it == 0:
        t.append(dt)
    else:
        t.append(t[-1] + dt)
    Tn = Tnplus1
    Psin = Psinplus1

    # Regenerate C
    C, rhsC = matrices.constructC(dxOpTemp=dxOpTemp, rhsDxOpTemp=rhsDxOpTemp, dyOpTemp=dyOpTemp,
                                  rhsDyOpTemp=rhsDyOpTemp, dlOpTemp=dlOpTemp, rhsDlOpTemp=rhsDlOpTemp, psi=Psin,
                                  dxOpPsi=dxOpPsi, dyOpPsi=dyOpPsi, sqrtRa=sqrtRa)

    # Solve for Tn+1
    if (useSuperLUFactorizationEq2):
        factor2 = sparse.linalg.factorized(sparse.csc_matrix(sparse.eye(nx * ny) + (dt / 2) * C))
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

    if ((it) % generateEvery == 0):
        if (generateAnimation):
            # Calculate speeds
            ux = grid.vecToField(dyOpPsi @ Psinplus1)
            uy = -grid.vecToField(dxOpPsi @ Psinplus1)
            maxSpeed = np.max(np.square(ux) + np.square(uy))
            maxSpeedArr.append(maxSpeed)

            # Calculate heat fluxes
            heatFluxConduction = np.sum(- grid.vecToField(dyOpTemp @ Tnplus1 - rhsDyOpTemp)[-10, :])
            heatFluxAdvection = sqrtRa * uy[-10, :] @ grid.vecToField(Tnplus1)[-10, :]
            totalHeatFlux = heatFluxAdvection + heatFluxConduction

            heatFluxConductionArr.append(heatFluxConduction)
            heatFluxAdvectionArr.append(heatFluxAdvection)
            totalHeatFluxArr.append(totalHeatFlux)

            # Plot the results
            # Velocity and temperature field
            ax1.quiver(np.linspace(0, 2, int(nx / quiverEveryNPoints)), np.linspace(0, 1, int(ny / quiverEveryNPoints)),
                       ux[::quiverEveryNPoints, ::quiverEveryNPoints], uy[::quiverEveryNPoints, ::quiverEveryNPoints])
            im = ax1.imshow((grid.vecToField(Tnplus1)), vmin=0, extent=[0, 2, 0, 1], aspect=1,
                            cmap=plt.get_cmap('magma'),
                            vmax=1, origin='lower')
            clrbr = plt.colorbar(im, ax=ax1)
            ax1.set_xlabel("x [distance]")
            ax1.set_ylabel("y [distance]")
            # ax1.title("t: %.2f, it: %i, Maximum speed: %.2e" % (t[-1], it, maxSpeed))

            # Maximum fluid velocity
            ax2.semilogy(t[::generateEvery], maxSpeedArr)
            ax2.legend(['Maximum fluid velocity'], fontsize=5, loc='lower right')
            ax2.set_xlabel("t [time]")
            ax2.set_ylabel("max V [speed]")
            ax2.set_xlim([0, (nt - generateEvery) * dt])
            ymax = 10
            ax2.set_ylim([1e-5, ymax if maxSpeed < ymax else maxSpeed])  # ternary statement, cool stuff

            # Heat fluxes
            ax3.plot(t[::generateEvery], heatFluxAdvectionArr)
            ax3.plot(t[::generateEvery], heatFluxConductionArr)
            ax3.plot(t[::generateEvery], totalHeatFluxArr, linestyle=':')
            ax3.legend(['Advective heat flux', 'Conductive heat flux', 'Total heat flux'], fontsize=5,
                       loc='center left')
            ax3.set_xlabel("t [time]")
            ax3.set_ylabel("q [heat flux]")
            ax3.set_xlim([0, (nt - generateEvery) * dt])
            ymax = totalHeatFluxArr[0] * 1.3
            ax3.set_ylim([0, ymax if totalHeatFlux < ymax else totalHeatFlux * 1.1])

            ax1.set_title("Temperature and velocity field", FontSize=7)
            ax2.set_title("Maximum fluid velocity over time", FontSize=7)
            ax3.set_title("Heat fluxes over time", FontSize=7)

            # Plot and redo!
            plt.savefig("fields/field%i.png" % (it / generateEvery))
            clrbr.remove()
            ax1.clear()
            ax2.clear()
            ax3.clear()
        print("time step:", it, end='\r')

end = time.time()
print("Runtime of time-marching: ", end - start, "seconds")

# And output figure(s)
if (generateFinal):
    ux = grid.vecToField(dyOpPsi @ Psinplus1)
    uy = grid.vecToField(dxOpPsi @ Psinplus1)
    maxSpeed = np.max(np.square(ux) + np.square(uy))
    plt.quiver(np.flipud(ux), -np.flipud(uy))
    plt.imshow(np.flipud(grid.vecToField(Tnplus1)), cmap=plt.get_cmap('magma'))
    plt.colorbar()
    plt.tight_layout()
    plt.title("Maximum speed: %.2f" % maxSpeed)
    plt.savefig("Convection-field.png")
