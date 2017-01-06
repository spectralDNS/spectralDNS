from numpy.polynomial import chebyshev as n_cheb
from numpy import zeros, pi

def points_and_weights(N, quad):
    if quad == "GL":
        points = -(n_cheb.chebpts2(N)).astype(float)
        weights = zeros(N)+pi/(N-1)
        weights[0] /= 2
        weights[-1] /= 2

    elif quad == "GC":
        points, weights = n_cheb.chebgauss(N)
        points = points.astype(float)
        weights = weights.astype(float)

    return points, weights
