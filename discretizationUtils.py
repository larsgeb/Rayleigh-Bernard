import numpy as np

import config


def index2to1(y, x):
    nx = config.nx
    ny = config.ny

    if (x < 0 or y < 0 or x >= nx or y >= ny):
        raise (IndexError)

    return (x * ny + y)


def index1to2y(k):
    ny = config.ny
    nx = config.nx

    if (k > nx * ny - 1 or k < 0):
        raise (IndexError)

    return int(k % ny)


def index1to2x(k):
    ny = config.ny
    nx = config.nx

    if (k > nx * ny - 1 or k < 0):
        raise (IndexError)

    return int(k / ny)


def fieldToVec(field):
    vec = np.ravel(field, order='F')
    vec.shape = (vec.size, 1)
    return vec


def vecToField(vec):
    nx = config.nx
    ny = config.ny
    return (np.reshape(vec, (ny, nx), order='F'))
