import scipy.sparse as sparse

import config
import discretizationUtils as grid


def makeSparseLaplaceKernel(dirichlet=None, neumann=None):
    ny = config.ny
    nx = config.nx
    dy = config.dy
    dx = config.dx
    Laplace = sparse.dok_matrix((nx * ny, nx * ny), dtype='float')
    rhsLaplace = sparse.dok_matrix((nx * ny, 1), dtype='float')

    for index, row in enumerate(Laplace):
        if (dirichlet.SparseIndices[grid.index1to2y(index), grid.index1to2x(index)] > 0):
            # print('Dirichlet value at pos %i %i' % (grid.index1to2y(index), grid.index1to2x(index)))
            Laplace[index, index] = 1
            rhsLaplace[index] = dirichlet.fixedVal[
                dirichlet.SparseIndices[grid.index1to2y(index), grid.index1to2x(index)] - 1]
        elif (neumann != None and neumann.SparseIndices[grid.index1to2y(index), grid.index1to2x(index)] > 0):
            # print('Neumann value at pos %i %i' % (grid.index1to2y(index), grid.index1to2x(index)))
            if (grid.index1to2x(index) == 0 or grid.index1to2x(index) == nx - 1):
                # Neumann in x-direction
                Laplace[index, index] = -(2.0 / dx ** 2 + 2.0 / dy ** 2)

                curX = grid.index1to2x(index)
                curY = grid.index1to2y(index)

                # This implements a Laplace operator on the defined grid
                try:
                    Laplace[index, grid.index2to1(curY - 1, curX)] = 1.0 / dy ** 2
                except (IndexError):
                    pass

                try:
                    Laplace[index, grid.index2to1(curY + 1, curX)] = 1.0 / dy ** 2
                except (IndexError):
                    pass

                # Which point in the x-sense don't we know?
                if (grid.index1to2x(index) == 0):
                    try:
                        Laplace[index, grid.index2to1(curY, curX + 1)] = 2 / dx ** 2
                        rhsLaplace[index] = - 2 * dx * neumann.fixedVal[
                            neumann.SparseIndices[grid.index1to2y(index), grid.index1to2x(index)] - 1]
                    except (IndexError):
                        pass
                else:
                    try:
                        Laplace[index, grid.index2to1(curY, curX - 1)] = 2 / dx ** 2
                        rhsLaplace[index] = 2 * dx * neumann.fixedVal[
                            neumann.SparseIndices[grid.index1to2y(index), grid.index1to2x(index)] - 1]
                    except (IndexError):
                        pass
            else:
                # Neumann in y-direction
                raise (NotImplementedError)
        else:
            # print('No Dirichlet value at pos %i %i' % (grid.index1to2y(index), grid.index1to2x(index)))
            Laplace[index, index] = -(2.0 / dx ** 2 + 2.0 / dy ** 2)

            curX = grid.index1to2x(index)
            curY = grid.index1to2y(index)

            # This implements a Laplace operator on the defined grid
            try:
                Laplace[index, grid.index2to1(curY - 1, curX)] = 1.0 / dy ** 2
            except (IndexError):
                # TODO make relevant exception, however this should never occur...
                pass

            try:
                Laplace[index, grid.index2to1(curY + 1, curX)] = 1.0 / dy ** 2
            except (IndexError):
                # TODO make relevant exception, however this should never occur...
                pass

            try:
                Laplace[index, grid.index2to1(curY, curX - 1)] = 1.0 / dx ** 2
            except (IndexError):
                # TODO make relevant exception, however this should never occur...
                pass

            try:
                Laplace[index, grid.index2to1(curY, curX + 1)] = 1.0 / dx ** 2
            except (IndexError):
                # TODO make relevant exception, however this should never occur...
                pass

    return Laplace, rhsLaplace


def makeSparsedxKernel(dirichlet=None, neumann=None):
    ny = config.ny
    nx = config.nx
    dy = config.dy
    dx = config.dx
    dxMat = sparse.dok_matrix((nx * ny, nx * ny), dtype='float')
    rhsDx = sparse.dok_matrix((nx * ny, 1), dtype='float')

    for index, row in enumerate(dxMat):
        if (neumann != None and neumann.SparseIndices[grid.index1to2y(index), grid.index1to2x(index)] > 0):
            # print('Neumann value at pos %i %i' % (grid.index1to2y(index), grid.index1to2x(index)))
            if (grid.index1to2x(index) == 0 or grid.index1to2x(index) == nx - 1):
                # Neumann in x-direction
                curX = grid.index1to2x(index)
                curY = grid.index1to2y(index)

                rhsDx[index] = - neumann.fixedVal[neumann.SparseIndices[curY, curX] - 1]

            else:
                # Neumann in y-direction
                raise (NotImplementedError)
        elif (grid.index1to2x(index) == 0):
            # print('Neumann condition absent, needed for central difference at boundary. Falling back to one sided Euler...')
            curX = grid.index1to2x(index)
            curY = grid.index1to2y(index)

            # Left point, let's try to implement a dirichlet value instead using one-sided euler
            if (dirichlet != None and dirichlet.SparseIndices[grid.index1to2y(index), grid.index1to2x(index)] > 0):
                try:
                    rhsDx[index] = dirichlet.fixedVal[dirichlet.SparseIndices[curY, curX] - 1] / (dx)
                except (IndexError):
                    # TODO make relevant exception, however this should never occur...
                    pass
            else:
                try:
                    dxMat[index, grid.index2to1(curY, curX)] = -1.0 / (dx)
                except (IndexError):
                    # TODO make relevant exception, however this should never occur...
                    pass

            # Right point
            try:
                dxMat[index, grid.index2to1(curY, curX + 1)] = 1.0 / (dx)
            except (IndexError):
                # TODO make relevant exception, however this should never occur...
                pass

        elif (grid.index1to2x(index) == nx - 1):
            # print('Neumann condition absent, needed for central difference at boundary. Falling back to one sided Euler...')
            curX = grid.index1to2x(index)
            curY = grid.index1to2y(index)

            try:
                dxMat[index, grid.index2to1(curY, curX - 1)] = -1.0 / (dx)
            except (IndexError):
                # TODO make relevant exception, however this should never occur...
                pass

            if (dirichlet != None and dirichlet.SparseIndices[grid.index1to2y(index), grid.index1to2x(index)] > 0):
                try:
                    rhsDx[index] = - dirichlet.fixedVal[dirichlet.SparseIndices[curY, curX] - 1] / (dx)
                except (IndexError):
                    # TODO make relevant exception, however this should never occur...
                    pass
            else:
                try:
                    dxMat[index, grid.index2to1(curY, curX)] = 1.0 / (dx)
                except (IndexError):
                    # TODO make relevant exception, however this should never occur...
                    pass

        else:
            curX = grid.index1to2x(index)
            curY = grid.index1to2y(index)

            # Left point
            try:
                dxMat[index, grid.index2to1(curY, curX - 1)] = -1.0 / (2.0 * dx)
            except (IndexError):
                # TODO make relevant exception, however this should never occur...
                pass
            # Right point
            try:
                dxMat[index, grid.index2to1(curY, curX + 1)] = 1.0 / (2.0 * dx)
            except (IndexError):
                # TODO make relevant exception, however this should never occur...
                pass

    return dxMat, rhsDx


def makeSparsedyKernel(dirichlet=None, neumann=None):
    ny = config.ny
    nx = config.nx
    dy = config.dy
    dx = config.dx
    dyMat = sparse.dok_matrix((nx * ny, nx * ny), dtype='float')
    rhsDy = sparse.dok_matrix((nx * ny, 1), dtype='float')

    for index, row in enumerate(dyMat):
        if (neumann != None and neumann.SparseIndices[grid.index1to2y(index), grid.index1to2x(index)] > 0):
            # print('Neumann value at pos %i %i' % (grid.index1to2y(index), grid.index1to2x(index)))
            if (grid.index1to2y(index) == 0 or grid.index1to2y(index) == ny - 1):
                # Neumann in y-direction
                curX = grid.index1to2x(index)
                curY = grid.index1to2y(index)

                rhsDy[index] = - neumann.fixedVal[neumann.SparseIndices[curY, curX] - 1]

            # else:
            # Neumann in x-direction
            # raise (NotImplementedError)

        elif (grid.index1to2y(index) == 0):
            # print(
            #     'Neumann condition absent, needed for central difference at boundary. Falling back to one sided Euler...')
            curX = grid.index1to2x(index)
            curY = grid.index1to2y(index)

            # Left point, let's try to implement a dirichlet value instead using one-sided euler
            if (dirichlet != None and dirichlet.SparseIndices[grid.index1to2y(index), grid.index1to2x(index)] > 0):
                try:
                    rhsDy[index] = dirichlet.fixedVal[dirichlet.SparseIndices[curY, curX] - 1] / (dy)
                except (IndexError):
                    # TODO make relevant exception, however this should never occur...
                    pass
            else:
                try:
                    dyMat[index, grid.index2to1(curY, curX)] = -1.0 / (dy)
                except (IndexError):
                    # TODO make relevant exception, however this should never occur...
                    pass

            # Right point
            try:
                dyMat[index, grid.index2to1(curY + 1, curX)] = 1.0 / (dy)
            except (IndexError):
                # TODO make relevant exception, however this should never occur...
                pass

        elif (grid.index1to2y(index) == ny - 1):
            # print(
            #     'Neumann condition absent, needed for central difference at boundary. Falling back to one sided Euler...')
            curX = grid.index1to2x(index)
            curY = grid.index1to2y(index)

            try:
                dyMat[index, grid.index2to1(curY - 1, curX)] = -1.0 / (dy)
            except (IndexError):
                # TODO make relevant exception, however this should never occur...
                pass

            if (dirichlet != None and dirichlet.SparseIndices[grid.index1to2y(index), grid.index1to2x(index)] > 0):
                try:
                    rhsDy[index] = - dirichlet.fixedVal[dirichlet.SparseIndices[curY, curX] - 1] / (dy)
                except (IndexError):
                    # TODO make relevant exception, however this should never occur...
                    pass
            else:
                try:
                    dyMat[index, grid.index2to1(curY, curX)] = 1.0 / (dy)
                except (IndexError):
                    # TODO make relevant exception, however this should never occur...
                    pass

        else:
            curX = grid.index1to2x(index)
            curY = grid.index1to2y(index)

            # Left point
            try:
                dyMat[index, grid.index2to1(curY - 1, curX)] = -1.0 / (2.0 * dy)
            except (IndexError):
                # TODO make relevant exception, however this should never occur...
                pass
            # Right point
            try:
                dyMat[index, grid.index2to1(curY + 1, curX)] = 1.0 / (2.0 * dy)
            except (IndexError):
                # TODO make relevant exception, however this should never occur...
                pass

    return dyMat, rhsDy


def plotKernel(mat, vec):
    import matplotlib.pyplot as plt
    f, axs = plt.subplots(1, 2)
    f.set_figheight(10)
    f.set_figwidth(10)
    im1 = axs[0].matshow(mat.todense().A)
    # plt.colorbar(im1,cax=axs[0])
    im2 = axs[1].matshow(vec.todense().A, aspect=0.1)
    # plt.colorbar(im2,cax=axs[1])
    plt.show()
