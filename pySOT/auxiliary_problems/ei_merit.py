import numpy as np
import scipy.spatial as scpspatial
from scipy.stats import norm


def ei_merit(X, surrogate, fX, XX=None, dtol=0):
    """Compute the expected improvement merit function.

    :param X: Points where to compute EI, of size n x dim
    :type X: numpy.array
    :param surrogate: Surrogate model object, must implement predict_std
    :type surrogate: object
    :param fX: Values at previously evaluated points, of size m x 1
    :type fX: numpy.array
    :param XX: Previously evaluated points, of size m x 1
    :type XX: numpy.array
    :param dtol: Minimum distance between evaluated and pending points
    :type dtol: float

    :return: Evaluate the expected improvement for points X
    :rtype: numpy.array of length X.shape[0]
    """
    mu, sig = surrogate.predict(X), surrogate.predict_std(X)
    gamma = (np.min(fX) - mu) / sig
    beta = gamma * norm.cdf(gamma) + norm.pdf(gamma)
    ei = sig * beta

    if dtol > 0:
        dists = scpspatial.distance.cdist(X, XX)
        dmerit = np.amin(dists, axis=1, keepdims=True)
        ei[dmerit < dtol] = 0.0
    return ei
