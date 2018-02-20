import numpy as np

import config


def index2to1(y, x):
    # TODO implement warnings outside of range
    nx = config.nx
    ny = config.ny

    if (x < 0 or y < 0 or x >= nx or y >= ny):
        raise (IndexError)

    return (x * ny + y)


def index1to2y(k):
    # TODO implement warnings outside of range
    ny = config.ny
    return int(k % ny)


def index1to2x(k):
    # TODO implement warnings outside of range
    ny = config.ny
    return int(k / ny)


def fieldToVec(field):
    return np.ravel(field, order='F')


def vecToField(vec):
    nx = config.nx
    ny = config.ny
    return (np.reshape(vec, (ny, nx), order='F'))
