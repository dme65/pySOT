import numpy as np


def median_capping(x):
    x = x.copy()
    medf = np.median(x)
    x[x > medf] = medf
    return x
