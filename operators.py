import config
import numpy as np
import scipy.sparse as sparse


def dyOp(nx, ny):
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


def dxOpTemp(nx, ny, bcNeuArray):
    # TODO
    dx = config.dx

    rhs = np.zeros((nx*ny,))
    rhs[0:ny] = - bcNeuArray[0]
    rhs[-ny:] = - bcNeuArray[1]

    plus1 = np.ones((ny,))
    mid = np.zeros((ny,))
    min1 = -np.ones((ny,))

    plus1 = np.tile(plus1, (nx - 1,))
    plus1[0:ny] = 0

    mid = np.tile(mid, (nx,))

    min1 = np.tile(min1, (nx - 1,))
    min1[-ny:-1] = 0

    return sparse.csr_matrix(sparse.diags([plus1[:-1], mid, min1[:-1]], [ny, 0, -ny]) / (2.0 * dx)),rhs

def dxOpStream(nx, ny):
    # TODO
    dx = config.dx

    plus1 = np.ones((ny,))
    mid = np.zeros((ny,))
    min1 = -np.ones((ny,))

    plus1 = np.tile(plus1, (nx - 1,))
    plus1[0:ny] = 2

    mid = np.tile(mid, (nx,))
    mid[0:ny] = -2
    mid[-ny:-1] = 2

    min1 = np.tile(min1, (nx - 1,))
    min1[-ny:-1] = -2

    return sparse.csr_matrix(sparse.diags([plus1[:-1], mid, min1[:-1]], [ny, 0, -ny]) / (2.0 * dx))


def dlOpStream(nx, ny):
    # Streamline Laplace operator
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


def dlOpTemp(nx, ny, bcDirArray, bcNeuArray):
    # Streamline Laplace operator
    dx = config.dx
    dy = config.dy

    rhs = np.zeros((ny * nx))
    for i in range(nx):
        rhs[0 + ny * i] = bcDirArray[0]
        rhs[ny * (i + 1) - 1] = bcDirArray[1]

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
    plus1x[1:ny - 1] = 2.0 / (dx ** 2)
    rhs[1:ny - 1] = rhs[1:ny - 1] + bcNeuArray[0] / dx
    min1x = np.tile(min1x, (nx - 1,))
    min1x[-ny:-1] = 2.0 / (dx ** 2)
    rhs[-ny + 1:-1] = rhs[-ny + 1:-1] - bcNeuArray[1] / dx

    # Assemble large scale y
    plus1y = np.tile(plus1y, (nx,))
    plus1y = plus1y[0:-1]
    # plus1y[0:ny] = 0
    # plus1y[-ny:] = 0
    min1y = np.tile(min1y, (nx,))
    min1y = min1y[1:]
    # min1y[0:ny] = 0
    # min1y[-ny:-1] = 0

    mid = np.tile(mid, (nx,))
    # mid[0:ny] = 1
    # mid[-ny:-1] = 1

    return sparse.csr_matrix(sparse.diags([mid, plus1y, min1y, plus1x, min1x], [0, 1, -1, ny, -ny])), rhs
