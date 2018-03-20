import config
import numpy as np
import scipy.sparse as sparse


def constructC(dxOpTemp, rhsDxOpTemp, dyOpTemp, rhsDyOpTemp, psi, dxOpPsi, dyOpPsi, sqrtRa):

    rhs = sqrtRa * (rhsDxOpTemp * (dyOpPsi * psi) - rhsDyOpTemp * (dxOpPsi * psi))

    return sqrtRa * (
            sparse.diags([(dxOpTemp * dyOpPsi * psi)[:, 0]], [0]) - sparse.diags([(dxOpPsi * psi)[:, 0]], [0])), rhs
    # return 0
