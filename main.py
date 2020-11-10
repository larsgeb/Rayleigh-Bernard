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
config.ny = 40
config.nx = 120
config.dy = 1.0 / (config.ny)
config.dx = 3.0 / (config.nx)
config.dt = 0.000125
config.nt = int(0.2 / config.dt)
dt = config.dt
nt = config.nt
ny = config.ny
nx = config.nx
dy = config.dy
dx = config.dx

# Use LU decomposition (for solving sparse system)
useSuperLUFactorizationEq1 = True  # doesn't do much as it needs to be done every loop
useSuperLUFactorizationEq2 = True  # Speeds up, quite a bit!

# Physical parameters (stream potential is 0 on all boundaries, hardcoded).
sqrtRa = 14
Tbottom = 1
Ttop = 0
tempDirichlet = np.array([np.ones((nx,)) * Tbottom, np.ones((nx,)) * Ttop])
tempNeumann = [0, 0]

# Use initial instability (also changes nt to 3000)
instability = True

# Plotting settings
generateAnimation = True
generateEvery = 10
generateFinal = False
qInt = 4  # What's the quiver vector interval? Want to plot every vector?
qScale = 40  # Scale the vectors?

# ---  No modifications below this point! --- #

assert (dx == dy)

# Initial conditions
startT = np.expand_dims(np.linspace(Tbottom, Ttop, ny), 1).repeat(nx, axis=1)
if instability:
    startT[int(ny / 2 - 3):int(ny / 2 + 3), int(nx / 2 - 3):int(nx / 2 + 3)] = 1.0
startT = grid.fieldToVec(startT)
startPsi = np.expand_dims(np.zeros(ny), 1).repeat(nx, axis=1)  # start with zeros for streamfunctions
startPsi = grid.fieldToVec(startPsi)

print("Buoyancy driven flow:\nRayleigh number: %.2f\ndt: %.2e\nnt: %i\ndx: %.2e\ndy: %.2e" % (
    sqrtRa * sqrtRa, dt, nt, dx, dy))
print("Critical CFL velocity: %.2e" % (dx / dt))

# Generate operators for differentials
dyOpPsi = op.dyOp()
dxOpPsi = op.dxOpStream()
dlOpPsi = op.dlOpStreamMod()
dlOpTemp, rhsDlOpTemp = op.dlOpTemp(bcDirArray=tempDirichlet, bcNeuArray=tempNeumann)
dxOpTemp, rhsDxOpTemp = op.dxOpTemp(bcNeuArray=tempNeumann)
dyOpTemp, rhsDyOpTemp = op.dyOpTemp(bcDirArray=tempDirichlet)

# This matrix is needed to only use the Non-Dirichlet rows in the del²psi operator. The other rows are basically
# ensuring psi_i,j = 0. What it does is it removes rows from a sparse matrix (unit matrix with some missing elements).
# PsiEliminator -> psiElim
psiElim = np.ones((nx * ny,))
psiElim[0:ny] = 0
psiElim[-ny:] = 0
psiElim[ny::ny] = 0
psiElim[ny - 1::ny] = 0
psiElim = sparse.csc_matrix(sparse.diags(psiElim, 0))

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
for it in range(
        nt + 1):  # Actually, we do one more than nt, but that's because otherwise the last (it = nt) wouldn't plot
    if it == 0:
        t.append(dt)
    else:
        t.append(t[-1] + dt)

    if ((it) % generateEvery == 0):
        if (generateAnimation):
            # calculate speeds
            ux = grid.vecToField(dyOpPsi @ Psinplus1) + 1e-40
            uy = -grid.vecToField(dxOpPsi @ Psinplus1) + 1e-40
            maxSpeed = 1e-99 + np.max(np.square(ux) + np.square(uy))
            maxSpeedArr.append(maxSpeed)

            # calculate heat fluxes
            heatFluxConduction = np.sum(- grid.vecToField(dyOpTemp @ Tnplus1 - rhsDyOpTemp)[int(ny / 2), :])
            heatFluxAdvection = sqrtRa * uy[int(ny / 2), :] @ grid.vecToField(Tnplus1)[int(ny / 2), :]
            totalHeatFlux = heatFluxAdvection + heatFluxConduction

            # accumulate values
            heatFluxConductionArr.append(heatFluxConduction)
            heatFluxAdvectionArr.append(heatFluxAdvection)
            totalHeatFluxArr.append(totalHeatFlux)

            # plot velocity and temperature field
            ax1.quiver(np.linspace(0, 2, int(nx / qInt)), np.linspace(0, 1, int(ny / qInt)), ux[::qInt, ::qInt],
                       uy[::qInt, ::qInt], pivot='mid', scale=qScale)
            im = ax1.imshow((grid.vecToField(Tnplus1)), vmin=0, vmax=1, extent=[0, 2, 0, 1], aspect=1,
                            cmap=plt.get_cmap('plasma'), origin='lower')
            clrbr = plt.colorbar(im, ax=ax1)
            clrbr.set_label('dimensionless temperature', rotation=270, labelpad=20)
            ax1.set_xlabel("x [distance]")
            ax1.set_ylabel("y [distance]")

            # plot maximum fluid velocity
            ymax = 10
            ax2.set_ylim([1e-5, ymax if np.max(maxSpeedArr) * (10 ** 0.1) < ymax else np.max(maxSpeedArr) * (
                    10 ** 0.1)])  # ternary statement, cool stuff
            ax2.semilogy(t[::generateEvery], maxSpeedArr)
            cflV = dx / dt
            ax2.semilogy([0, nt * dt], [cflV, cflV], linestyle=":")
            ax2.legend(['Maximum fluid velocity', 'CFL limit %.2f' % cflV], fontsize=5, loc='lower right')
            ax2.set_xlabel("t [time]")
            ax2.set_ylabel("max V [speed]")
            ax2.set_xlim([0, nt * dt])
            ax2.set_ylim([0.01, ax2.set_ylim()[1]])

            # plot heat fluxes
            ax3.semilogy(t[::generateEvery], heatFluxAdvectionArr)
            ax3.semilogy(t[::generateEvery], heatFluxConductionArr)
            ax3.semilogy(t[::generateEvery], totalHeatFluxArr, linestyle=':')
            ax3.legend(['Advective heat flux', 'Conductive heat flux', 'Total heat flux'], fontsize=5,
                       loc='lower right')
            ax3.set_xlabel("t [time]")
            ax3.set_ylabel("q [heat flux]")
            ax3.set_xlim([0, nt * dt])
            ymax = totalHeatFluxArr[0] * 1.3
            ax3.set_ylim([0.01, ymax if np.max(totalHeatFluxArr) * 1.1 < ymax else np.max(totalHeatFluxArr) * 1.1])

            # Plot titles
            ax1.set_title("Temperature and velocity field\nstep = %i, dt = %.2e, t = %.2f" % (it, dt, t[-1] - dt),
                          fontsize=7)
            ax2.set_title("Maximum fluid velocity over time", fontsize=7)
            ax3.set_title("Heat fluxes over time", fontsize=7)

            # Plot and redo!
            plt.savefig("simulations/field%05i.png" % (it / generateEvery))
            clrbr.remove()
            ax1.clear()
            ax2.clear()
            ax3.clear()
            print("plotted time step:", it, end='\r')

    # reset time level for fields
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
    Tnplus1[0::ny] = np.expand_dims(tempDirichlet[0], 1)
    Tnplus1[ny - 1::ny] = np.expand_dims(tempDirichlet[1], 1)

    # Solve for Psi n+1
    if (useSuperLUFactorizationEq1):
        Psinplus1 = factor1(- sqrtRa * psiElim @ (dxOpTemp @ Tnplus1 - rhsDxOpTemp))
    else:
        Psinplus1 = sparse.linalg.spsolve(dlOpPsi, - sqrtRa * psiElim @ (dxOpTemp @ Tnplus1 - rhsDxOpTemp))

    # Enforcing column vector
    Psinplus1.shape = (nx * ny, 1)
    # Enforcing Dirichlet
    Psinplus1[0::ny] = 0
    Psinplus1[ny - 1::ny] = 0
    Psinplus1[0:ny] = 0
    Psinplus1[-ny:] = 0

end = time.time()
print("\nRuntime of time-marching: %.2e seconds" % (end - start))

# And output figure(s)
# if (generateFinal):
#     ux = grid.vecToField(dyOpPsi @ Psinplus1)
#     uy = grid.vecToField(dxOpPsi @ Psinplus1)
#     maxSpeed = np.max(np.square(ux) + np.square(uy))
#     plt.quiver(np.flipud(ux), -np.flipud(uy))
#     plt.imshow(np.flipud(grid.vecToField(Tnplus1)), cmap=plt.get_cmap('magma'))
#     plt.colorbar()
#     plt.tight_layout()
#     plt.title("Maximum speed: %.2f" % maxSpeed)
#     plt.savefig("Convection-field.png")
