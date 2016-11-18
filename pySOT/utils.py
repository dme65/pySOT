"""
.. module:: utils
   :synopsis: Help functions for pySOT

.. moduleauthor:: David Eriksson <dme65@cornell.edu>

:Module: utils
:Author: David Eriksson <dme65@cornell.edu>

"""

import numpy as np


def to_unit_box(x, data):
    """Maps a set of points to the unit box

    :param x: Points to be mapped to the unit box, of size npts x dim
    :type x: numpy.array
    :param data: Optimization problem, needs to have attributes xlow and xup
    :type data: Object
    :return: Points mapped to the unit box
    :rtype: numpy.array
    """

    return (np.copy(x) - data.xlow) / (data.xup - data.xlow)


def from_unit_box(x, data):
    """Maps a set of points from the unit box to the original domain

    :param x: Points to be mapped from the unit box, of size npts x dim
    :type x: numpy.array
    :param data: Optimization problem, needs to have attributes xlow and xup
    :type data: Object
    :return: Points mapped to the original domain
    :rtype: numpy.array
    """

    return data.xlow + (data.xup - data.xlow) * np.copy(x)


def unit_rescale(xx):
    """Shift and rescale elements of a vector to the unit interval

    :param xx: Vector that should be rescaled to the unit interval
    :type xx: numpy.array
    :return: Vector scaled to the unit interval
    :rtype: numpy.array
    """

    xmax = np.amax(xx)
    xmin = np.amin(xx)
    if xmax == xmin:
        return np.ones(xx.shape)
    else:
        return (xx-xmin)/(xmax-xmin)


def round_vars(data, x):
    """Round integer variables to closest integer that is still in the domain

    :param data: Optimization problem object
    :type data: Object
    :param x: Set of points, of size npts x dim
    :type x: numpy.array
    :return: The set of points with the integer variables
        rounded to the closest integer in the domain
    :rtype: numpy.array
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


def check_opt_prob(obj):
    """Routine for checking that an implementation of the optimization problem
    follows the standard. This method checks everything, but can't make
    sure that the objective function and constraint methods return values
    of the correct type since this would involve actually evaluating the
    objective function which isn't feasible when the evaluations are
    expensive. If some test fails, an exception is raised through assert.

    :param obj: Optimization problem
    :type obj: Object
    :raise AttributeError: If object doesn't follow the pySOT standard
    """

    if not hasattr(obj, "dim"):
        raise AttributeError("Problem dimension required")
    if not hasattr(obj, "xlow"):
        raise AttributeError("Numpy array of lower bounds required")
    if not isinstance(obj.xlow, np.ndarray):
        raise AttributeError("Numpy array of lower bounds required")
    if not hasattr(obj, "xup"):
        raise AttributeError("Numpy array of upper bounds required")
    if not isinstance(obj.xup, np.ndarray):
        raise AttributeError("Numpy array of upper bounds required")
    if not hasattr(obj, "integer"):
        raise AttributeError("Integer variables must be specified")
    if len(obj.integer) > 0:
        if not isinstance(obj.integer, np.ndarray):
            raise AttributeError("Integer variables must be specified")
    else:
        if not(isinstance(obj.integer, np.ndarray) or
               isinstance(obj.integer, list)):
            raise AttributeError("Integer variables must be specified")
    if not hasattr(obj, "continuous"):
        raise AttributeError("Continuous variables must be specified")
    if len(obj.continuous) > 0:
        if not isinstance(obj.continuous, np.ndarray):
            raise AttributeError("Continuous variables must be specified")
    else:
        if not(isinstance(obj.continuous, np.ndarray) or
               isinstance(obj.continuous, list)):
            raise AttributeError("Continuous variables must be specified")

    # Check for logical errors
    if not (isinstance(obj.dim, int) and obj.dim > 0):
        raise AttributeError("Problem dimension must be a positive integer.")
    if not(len(obj.xlow) == obj.dim and
            len(obj.xup) == obj.dim):
        raise AttributeError("Incorrect size for xlow and xup")
    if not(all(obj.xlow[i] < obj.xup[i] for i in range(obj.dim))):
        raise AttributeError("Lower bounds must be below upper bounds.")
    if len(obj.integer) > 0:
        if not(np.amax(obj.integer) < obj.dim and np.amin(obj.integer) >= 0):
            raise AttributeError("Integer variable index can't exceed "
                                 "number of dimensions or be negative")
    if len(obj.continuous) > 0:
        if not(np.amax(obj.continuous) < obj.dim and
               np.amin(obj.continuous) >= 0):
            raise AttributeError("Continuous variable index can't exceed "
                                 "number of dimensions or be negative")
    if not(len(np.intersect1d(obj.continuous, obj.integer)) == 0):
        raise AttributeError("A variable can't be both an integer and continuous")
    if not(len(obj.continuous)+len(obj.integer) == obj.dim):
        raise AttributeError("All variables must be either integer or continuous")


def progress_plot(controller, title='', interactive=False):
    """Makes a progress plot from a POAP controller

    This method depends on matplotlib and will terminate if matplotlib.pyplot
    is unavailable.

    :param controller: POAP controller object
    :type controller: Object
    :param title: Title of the plot
    :type title: string
    :param interactive: True if the plot should be interactive
    :type interactive: bool
    """

    try:
        import matplotlib.pyplot as plt
        plotting_on = True
    except:
        plotting_on = False
        pass

    if not plotting_on:
        print("Failed to import matplotlib.pyplot, aborting....")
        return

    # Extract function values from the controller, ignoring crashed evaluations
    fvals = np.array([o.value for o in controller.fevals if o.value is not None])

    plt.figure()
    if interactive:
        plt.ion()
    plt.plot(np.arange(0, fvals.shape[0]), fvals, 'bo')  # Points
    plt.plot(np.arange(0, fvals.shape[0]), np.minimum.accumulate(fvals),
             'r-', linewidth=4.0)  # Best value found

    # Set limits
    ymin = np.min(fvals)
    ymax = np.max(fvals)
    plt.ylim(ymin - 0.1 * (ymax - ymin), ymax + 0.1 * (ymax - ymin))

    plt.xlabel('Evaluations')
    plt.ylabel('Function Value')
    plt.title(title)
    plt.show()
