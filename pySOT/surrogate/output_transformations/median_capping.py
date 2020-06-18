import numpy as np


def median_capping(x):
    """Replace values above the median by the median.

    :param x: Array to be transformed
    :type x: numpy.array

    :return: x
    :rtype: numpy.array
    """
    x = x.copy()
    medf = np.median(x)
    x[x > medf] = medf
    return x
