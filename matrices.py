import config
import numpy as np
import scipy.sparse as sparse


def constructC(dxOpTemp, rhsDxOpTemp, dyOpTemp, rhsDyOpTemp, dlOpTemp, rhsDlOpTemp, psi, dxOpPsi, dyOpPsi, sqrtRa):

    dxPsi = sparse.diags((dxOpPsi @ psi)[:, 0], 0)
    dyPsi = sparse.diags((dyOpPsi @ psi)[:, 0], 0)

    C = sqrtRa * (dyPsi @ dxOpTemp - dxPsi @ dyOpTemp ) - dlOpTemp

    rhsC = - rhsDlOpTemp + sqrtRa * (dyPsi @ rhsDxOpTemp- dyPsi @ rhsDyOpTemp)

    return C,rhsC
