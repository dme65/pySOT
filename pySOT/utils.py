import numpy as np


def to_unit_box(x, data):
    return (np.copy(x) - data.xlow) / (data.xup - data.xlow)


def from_unit_box(x, data):
        return data.xlow + (data.xup - data.xlow) * np.copy(x)


def unit_rescale(xx):
    """Shift and rescale elements of a vector to the unit interval
    """
    xmax = np.amax(xx)
    xmin = np.amin(xx)
    if xmax == xmin:
        return np.ones(xx.shape)
    else:
        return (xx-xmin)/(xmax-xmin)


def round_vars(data, x):
    """Round integer variables to closest integer
    """
    if len(data.integer) > 0:
        # Round the original ranged integer variables
        x[:, data.integer] = np.round(x[:, data.integer])
        # Make sure we don't violate the bound constraints
        for i in data.integer:
            ind = np.where(x[:, i] < data.xlow[i])
            x[ind, i] += 1
            ind = np.where(x[:, i] > data.xup[i])
            x[ind, i] -= 1
    return x