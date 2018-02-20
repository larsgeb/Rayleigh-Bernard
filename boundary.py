import numpy as np
import scipy.sparse as sparse

import config


class BoundaryValue():
    def load(self):
        self.ny = config.ny
        self.nx = config.nx
        self.dy = config.dy
        self.dx = config.dx

    def __init__(self, x0=None, xend=None, y0=None, yend=None):
        """
        Setting Dirichlet Boundary conditions. The hierarchy of these is x0, xend, y0, yend
        This means that if both boundaries are not equal, the first in this list gets priority.
        :param x0: value at x0, default is None
        :param xend: value at xend, default is None
        :param y0: value at y0, default is None
        :param yend: value at yend, default is None
        """
        self.load()

        row = np.array([])
        col = np.array([])
        data = np.array([])
        self.fixedVal = np.array([0, 0, 0, 0])

        if (x0 != None):
            row = np.concatenate((row, np.arange(self.ny)))
            col = np.concatenate((col, np.zeros(self.ny)))
            data = np.concatenate((data, np.ones(self.ny)))
            self.fixedVal[0] = x0
        if (xend != None):
            row = np.concatenate((row, np.arange(self.ny)))
            col = np.concatenate((col, (self.nx - 1) * np.ones(self.ny)))
            data = np.concatenate((data, 2 * np.ones(self.ny)))
            self.fixedVal[1] = xend
        if (y0 != None):
            y0start = 0
            numy0 = self.nx
            # x0 boundary override
            if (x0 != None):
                numy0 = numy0 - 1
                y0start = y0start + 1
            # xend boundary override
            if (xend != None):
                numy0 = numy0 - 1

            row = np.concatenate((row, np.zeros(numy0)))
            col = np.concatenate((col, y0start + np.arange(numy0)))
            data = np.concatenate((data, 3 * np.ones(numy0)))
            self.fixedVal[2] = y0
        if (yend != None):
            yendstart = 0
            numyend = self.nx
            # x0 boundary override
            if (x0 != None):
                numyend = numyend - 1
                yendstart = yendstart + 1
            # xend boundary override
            if (xend != None):
                numyend = numyend - 1

            row = np.concatenate((row, (self.ny - 1) * np.ones(numyend)))
            col = np.concatenate((col, yendstart + np.arange(numyend)))
            data = np.concatenate((data, 4 * np.ones(numyend)))
            self.fixedVal[3] = yend
        self.SparseIndices = sparse.dok_matrix(sparse.coo_matrix((data, (row, col)), shape=(self.ny, self.nx)),
                                               dtype='int')
        self.Values = [x0, xend, y0, yend]
        # print('Sparsity: ', self.DirichletSparseMat.getnnz() / (self.nx * self.ny))


class Dirichlet(BoundaryValue):
    def __init__(self, x0=None, xend=None, y0=None, yend=None):
        super().__init__(x0, xend, y0, yend)
        self.type = "Dirichlet"


class Neumann(BoundaryValue):
    def __init__(self, x0=None, xend=None, y0=None, yend=None):
        if (y0 != None or yend != None):
            # No need in this project
            raise (NotImplementedError)

        super().__init__(x0, xend, y0, yend)
        self.type = "Neumann"
