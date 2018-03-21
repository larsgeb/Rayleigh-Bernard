import config
import numpy as np
import scipy.sparse as sparse


# TODO: Higher order FD schemes
# A good countermeasure for the low order approximations at the boundaries would be to implement higher order
# single sided formulas using three or more gridpoints.

# General operators (not useful for solving)
def dyOp():
    """Create discrete operator on 2d grid for d( . ) / dy using central and one-side difference formulas. The one-sided
    formulas are only used at the boundaries. Doesn't use any boundary conditions (and is therefore unused in the
    project).

    Status: finished on 21 March.

    :return: scipy.CSR_matrix containing linear operator
    """
    nx = config.nx
    ny = config.ny
    dy = config.dy

    plus1 = np.ones((ny,))
    plus1[-1] = 0
    plus1[0] = 2
    mid = np.zeros((ny,))
    mid[0] = -2
    mid[-1] = 2
    min1 = -np.ones((ny,))
    min1[-1] = 0
    min1[-2] = -2

    plus1 = np.tile(plus1, (nx,))
    mid = np.tile(mid, (nx,))
    min1 = np.tile(min1, (nx,))

    return sparse.csr_matrix(sparse.diags([plus1[:-1], mid, min1[:-1]], [1, 0, -1]) / (2.0 * dy))


# Temperature operators
def dyOpTemp(bcDirArray):
    """Create discrete operator on 2d grid for dT / dy using central and one-side difference formulas. The one-sided
    formulas are only used at the boundaries, and Dirichlet boundary conditions are used to eliminate nodes from the
    LHS. To calculate the dT / dy field one would perform A @ T - rhs.

    Status: finished on 21 March.

    :return: tuple of (scipy.CSR_matrix containing linear operator, numpy.ndarray of rhs vector)
    """
    nx = config.nx
    ny = config.ny
    dy = config.dy

    # First construct the right hand side from one-sided difference formulas for y = 0 & y = ymax and the Dirichlet BC.
    rhs = np.zeros((nx * ny,))
    for i in range(nx):
        rhs[0 + ny * i] = bcDirArray[0] / dy  # y = 0
        rhs[ny * (i + 1) - 1] = - bcDirArray[1] / dy  # y = ymax
    rhs = np.expand_dims(rhs, 1)  # Make it an explicit column vector

    # We use the central difference formula for all the inner derivatives ...
    plus1 = np.ones((ny,))
    mid = np.zeros((ny,))
    min1 = -np.ones((ny,))

    # ... and again the one-sided difference formulas for y = 0 & y = ymax
    plus1[-1] = 0
    plus1[0] = 2
    min1[-1] = 0
    min1[-2] = -2

    # Repeat along x = 0, 1, 2, ...
    plus1 = np.tile(plus1, (nx,))
    mid = np.tile(mid, (nx,))
    min1 = np.tile(min1, (nx,))

    # Construct a sparse matrix from diagonals
    A = sparse.csr_matrix(sparse.diags([plus1[:-1], mid, min1[:-1]], [1, 0, -1]) / (2.0 * dy)), rhs

    return A, rhs


def dxOpTemp(bcNeuArray):
    """Create discrete operator on 2d grid for dT / dx using central and one-side difference formulas. The one-sided
    formulas are only used at the boundaries, and Dirichlet boundary conditions are used to eliminate nodes from the
    LHS. To calculate the dT / dx field one would perform A @ T - rhs.

    Status: finished on 20 March.

    :return: tuple of (scipy.CSR_matrix containing linear operator, numpy.ndarray of rhs vector)
    """
    nx = config.nx
    ny = config.ny
    dx = config.dx

    # TODO Check this
    rhs = np.zeros((nx * ny,))
    rhs[0:ny] = - bcNeuArray[0]
    rhs[-ny:] = - bcNeuArray[1]
    rhs = np.expand_dims(rhs, 1)  # Make it an explicit column vector

    plus1 = np.ones((ny,))
    mid = np.zeros((ny,))
    min1 = -np.ones((ny,))

    plus1 = np.tile(plus1, (nx - 1,))
    plus1[0:ny] = 0

    mid = np.tile(mid, (nx,))

    min1 = np.tile(min1, (nx - 1,))
    min1[-ny:] = 0

    A = sparse.csr_matrix(sparse.diags([plus1[:], mid, min1[:]], [ny, 0, -ny]) / (2.0 * dx))

    return A, rhs


def dlOpTemp(bcDirArray, bcNeuArray):
    """Create discrete operator on 2d grid for d²T / dx² + d²T / dy² using central and one-side difference formulas.
    The one-sided formulas are only used at the boundaries, and Dirichlet boundary conditions are used to eliminate
    nodes from the LHS. Neumann conditions are used to simplify calculation of d²T/dx². To calculate the dT / dx field
    one would perform A @ T - rhs.

    Status: finished on 21 March.

    :return: tuple of (scipy.CSR_matrix containing linear operator, numpy.ndarray of rhs vector)
    """
    nx = config.nx
    ny = config.ny
    dx = config.dx
    dy = config.dy

    rhs = np.zeros((nx * ny,))
    rhs.shape = (nx * ny, 1)

    # This is the center of the stencil in centered difference formula
    mid = -2 * np.ones((ny,)) / (dx ** 2) - 2 * np.ones((ny,)) / (dy ** 2)
    # But the points at y = 0 and y = ymax are using a one-sided difference
    mid[0] = -2 / (dx ** 2) + 1 / (dy ** 2)
    mid[-1] = mid[0]
    mid = np.tile(mid, (nx,))
    # And for x=0 and x=xmax we also have the modified formula for the x coordinate (note that the formula actually is
    # very similar as the centered difference, although accuracy is lower.)
    rhs[0:ny] = bcNeuArray[0] / (0.5 * dx)
    rhs[-ny:] = -bcNeuArray[1] / (0.5 * dx)

    # Two sided difference formula for d/dy to next point
    plus1y = np.ones((ny,)) / (dy ** 2)
    plus1y[0] = -2 / (dy ** 2)  # one sided
    plus1y[ny - 1] = 0  # not part of the scheme
    plus1y = np.tile(plus1y, (nx,))
    plus1y = plus1y[:-1]

    # Two sided difference formula for d/dy to previous point
    min1y = np.ones((ny,)) / (dy ** 2)
    min1y[-2] = -2 / (dy ** 2)  # one side
    min1y[-1] = 0  # not part of the scheme
    min1y = np.tile(min1y, (nx,))
    min1y = min1y[:-1]

    plus2y = np.zeros((ny,))
    plus2y[0] = 1 / (dy ** 2)
    plus2y = np.tile(plus2y, (nx,))
    plus2y = plus2y[:-2]

    min2y = np.zeros((ny,))
    min2y[-3] = 1 / (dy ** 2)
    min2y = np.tile(min2y, (nx,))
    min2y = min2y[:-2]

    plus1x = np.ones((ny,)) / (dx ** 2)
    plus1x = np.tile(plus1x, (nx - 1,))
    plus1x[:ny] = 2 / (dx ** 2)

    min1x = np.ones((ny,)) / (dx ** 2)
    min1x = np.tile(min1x, (nx - 1,))
    min1x[-ny:] = 2 / (dx ** 2)

    A = np.diag(mid) + np.diag(min1y, -1) + np.diag(plus1y, 1) + np.diag(plus2y, 2) + np.diag(min2y, -2) \
        + np.diag(plus1x, ny) + np.diag(min1x, -ny)

    # Now we need to construct the RHS for the Dirichlet points
    for ix in range(nx):
        rhsPart = A[:, ix * ny] * bcDirArray[0]
        rhsPart.shape = (nx * ny, 1)
        rhs = rhs - np.copy(rhsPart)
        A[:, ix * ny] = 0
        rhsPart = A[:, (ix + 1) * ny - 1] * bcDirArray[1]
        rhsPart.shape = (nx * ny, 1)
        rhs = rhs - np.copy(rhsPart)
        A[:, (ix + 1) * ny - 1] = 0

    return A, rhs


# def dlOpTempLegacy(bcDirArray, bcNeuArray):
#     # TODO
#     # Streamline Laplace operator
#     nx = config.nx
#     ny = config.ny
#     dx = config.dx
#     dy = config.dy
#
#     rhs = np.zeros((nx * ny,))
#     rhs.shape = (nx * ny, 1)
#
#     mid = -2 * np.ones((ny,)) / (dy ** 2) - 2 * np.ones((ny,)) / (dx ** 2)
#     mid[0] = 0
#     mid[-1] = 0
#     mid = np.tile(mid, (nx,))
#     # These are the one-sided differences for d²T/dx²
#     mid[1:ny - 1] = -2 / (dy ** 2) - 1 / (2 * dx ** 2)
#     mid[-ny + 1:-1] = -2 / (dy ** 2) - 1 / (2 * dx ** 2)
#
#     plus1x = np.ones((ny,)) / (dx ** 2)
#     plus1x[0] = 0
#     plus1x[-1] = 0
#     plus1x = np.tile(plus1x, (nx - 1,))
#     plus1x[0:ny] = 0
#
#     min1x = np.ones((ny,)) / (dx ** 2)
#     min1x[0] = 0
#     min1x[-1] = 0
#     min1x = np.tile(min1x, (nx - 1,))
#     min1x[-ny:] = 0
#     # Associated RHS
#     rhs[ny::ny] = -bcDirArray[0] / (dx ** 2)
#     rhs[ny +::ny] = -bcDirArray[0] / (dx ** 2)
#
#     # Check
#     plus2x = np.zeros((ny,))
#     plus2x = np.tile(plus2x, (nx - 2,))
#     plus2x[1:ny - 1] = 1 / (2 * dx ** 2)
#     # Associated RHS
#     rhs[1:ny - 1] = rhs[1:ny - 1] + bcNeuArray[0] / dx
#
#     # Check
#     min2x = np.zeros((ny,))
#     min2x = np.tile(min2x, (nx - 2,))
#     min2x[-ny:] = 1 / (2 * dx ** 2)
#     # Associated RHS
#     rhs[-ny + 1:-1] = rhs[-ny + 1:-1] - bcNeuArray[1] / dx
#
#     min1y = np.ones((ny,)) / (dy ** 2)
#     min1y[0] = 0  # Dirichlet
#     min1y[-2] = -2 / (dy ** 2)  # One-side difference
#     min1y[-1] = 0  # Not part of current column
#     min1y = np.tile(min1y, (nx,))[0:-1]
#     # Associated RHS
#     rhs[1::ny] = rhs[1::ny] - bcDirArray[0] / (dy ** 2)
#
#     plus1y = np.ones((ny,)) / (dy ** 2)
#     plus1y[0] = -2 / (dy ** 2)  # One-side difference
#     plus1y[-2] = 0  # Dirichlet
#     plus1y[-1] = 0  # Not part of current column
#     plus1y = np.tile(plus1y, (nx,))[0:-1]
#     # Associated RHS
#     rhs[ny - 2::ny] = rhs[ny - 2::ny] - bcDirArray[1] / (dy ** 2)
#
#     min2y = np.zeros((ny,))
#     min2y[1] = 1 / (dy ** 2)
#     min2y = np.tile(min2y, (nx,))[0:-2]
#
#     plus2y = np.zeros((ny,))
#     plus2y[0] = 1 / (dy ** 2)
#     plus2y = np.tile(plus2y, (nx,))[0:-2]
#
#     A = np.diag(mid) + np.diag(min1y, -1) + np.diag(min2y, -2) + np.diag(plus1y, 1) + np.diag(plus2y, 2) + np.diag(
#         plus1x, ny) + np.diag(
#         min1x, -ny) + \
#         np.diag(plus2x, ny * 2) + np.diag(min2x, -ny * 2)
#
#     return A, rhs


# Streamfunction operators
def dxOpStream():
    """Create discrete operator on 2d grid for dPsi / dx using central and one-side difference formulas. The one-sided
    formulas are only used at the boundaries, and Dirichlet boundary conditions are used to eliminate nodes from the
    LHS. To calculate the dPsi / dx field one would perform A @ T. I omitted including the Dirichlet conditions because
    this matrix is never used to solve for Psi.

    Status: finished on 21 March.

    :return: scipy.CSR_matrix containing linear operator
    """
    nx = config.nx
    ny = config.ny
    dx = config.dx

    plus1 = np.ones((ny,))
    mid = np.zeros((ny,))
    min1 = -np.ones((ny,))

    plus1 = np.tile(plus1, (nx - 1,))
    plus1[0:ny] = 2

    mid = np.tile(mid, (nx,))
    mid[0:ny] = -2
    mid[-ny:] = 2

    min1 = np.tile(min1, (nx - 1,))
    min1[-ny:] = -2

    A = sparse.csr_matrix(sparse.diags([plus1[:], mid, min1[:]], [ny, 0, -ny]) / (2.0 * dx))

    return A


def dyOpStream():
    """Create discrete operator on 2d grid for dPsi / dy using central and one-side difference formulas. The one-sided
    formulas are only used at the boundaries, and Dirichlet boundary conditions are used to eliminate nodes from the
    LHS. To calculate the dPsi / dy field one would perform A @ T. I omitted including the Dirichlet conditions because
    this matrix is never used to solve for Psi. It is of course then identical to dyOp().

    Status: finished on 21 March.

    :return: scipy.CSR_matrix containing linear operator
    """
    nx = config.nx
    ny = config.ny
    dy = config.dy

    plus1 = np.ones((ny,))
    plus1[-1] = 0
    plus1[0] = 2
    mid = np.zeros((ny,))
    mid[0] = -2
    mid[-1] = 2
    min1 = -np.ones((ny,))
    min1[-1] = 0
    min1[-2] = -2

    plus1 = np.tile(plus1, (nx,))
    mid = np.tile(mid, (nx,))
    min1 = np.tile(min1, (nx,))

    A = sparse.csr_matrix(sparse.diags([plus1[:-1], mid, min1[:-1]], [1, 0, -1]) / (2.0 * dy))

    return A


def dlOpStreamMod():
    """Create discrete operator on 2d grid for d²Psi / dy² using central and Dirichlet boundary conditions. It is not
    the traditional operator, as some entries are modified to simplify the solving of equation 1 from the project.

        Status: finished on 21 March.

        :return: scipy.CSR_matrix containing linear operator
        """
    nx = config.nx
    ny = config.ny
    dx = config.dx
    dy = config.dy

    mid = -2 * np.ones((ny,)) / (dx ** 2) + -2 * np.ones((ny,)) / (dy ** 2)
    mid[0] = 1
    mid[-1] = 1

    plus1x = np.ones((ny,)) / (dx ** 2)
    plus1x[0] = 0
    plus1x[-1] = 0
    min1x = np.ones((ny,)) / (dx ** 2)
    min1x[0] = 0
    min1x[-1] = 0

    plus1y = np.ones((ny,)) / (dy ** 2)
    plus1y[0] = 0
    plus1y[-1] = 0
    min1y = np.ones((ny,)) / (dy ** 2)
    min1y[0] = 0
    min1y[-1] = 0

    # Assemble large scale x
    plus1x = np.tile(plus1x, (nx - 1,))
    plus1x[0:ny] = 0
    min1x = np.tile(min1x, (nx - 1,))
    # min1x[0:ny] = 0
    min1x[-ny:-1] = 0

    # Assemble large scale y
    plus1y = np.tile(plus1y, (nx,))
    plus1y = plus1y[0:-1]
    plus1y[0:ny] = 0
    plus1y[-ny:] = 0
    min1y = np.tile(min1y, (nx,))
    min1y = min1y[1:]
    min1y[0:ny] = 0
    min1y[-ny:-1] = 0

    mid = np.tile(mid, (nx,))
    mid[0:ny] = 1
    mid[-ny:-1] = 1

    return sparse.csr_matrix(sparse.diags([mid, plus1y, min1y, plus1x, min1x], [0, 1, -1, ny, -ny]))
