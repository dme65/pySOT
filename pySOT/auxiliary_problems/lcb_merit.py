import numpy as np
import scipy.spatial as scpspatial


def lcb_merit(X, surrogate, fX, XX=None, dtol=0.0, kappa=2.0):
    """Compute the lcb merit function.

    :param X: Points where to compute LCB, of size n x dim
    :type X: numpy.array
    :param surrogate: Surrogate model object, must implement predict_std
    :type surrogate: object
    :param fX: Values at previously evaluated points, of size m x 1
    :type fX: numpy.array
    :param XX: Previously evaluated points, of size m x 1
    :type XX: numpy.array
    :param dtol: Minimum distance between evaluated and pending points
    :type dtol: float
    :param kappa: Constant in front of standard deviation
        Default: 2.0
    :type kappa: float

    :return: Evaluate the lower confidence bound for points X
    :rtype: numpy.array of length X.shape[0]
    """
    mu, sig = surrogate.predict(X), surrogate.predict_std(X)
    lcb = mu - kappa * sig

    if dtol > 0:
        dists = scpspatial.distance.cdist(X, XX)
        dmerit = np.amin(dists, axis=1, keepdims=True)
        lcb[dmerit < dtol] = np.inf
    return lcb
