import warnings

import numpy as np
import pyDOE2 as pydoe

from .experimental_design import ExperimentalDesign, _expdes_dist


class LatinHypercube(ExperimentalDesign):
    """Latin Hypercube experimental design.

    :param dim: Number of dimensions
    :type dim: int
    :param num_pts: Number of desired sampling points
    :type num_pts: int
    :param criterion: Previously passed to pyDOE, now deprecated
    :type criterion: string
    :param iterations: Number of designs to choose from
    :type iterations: int

    :ivar dim: Number of dimensions
    :ivar num_pts: Number of points in the experimental design
    :ivar iterations: Number of points in the experimental design
    """

    def __init__(self, dim, num_pts, criterion=None, iterations=1000):
        if criterion is not None:
            warnings.warn("Criterion is deprecated and will be removed.")
        self.dim = dim
        self.num_pts = num_pts
        self.iterations = iterations

    def generate_points(self, lb=None, ub=None, int_var=None):
        """Generate a new experimental design.

        You can specify lb, ub, int_var to have the design mapped to a
        specific domain. These inputs are ignored if one of lb
        or ub is None. The design is generated in [0, 1]^d in this case.

        :param lb: Lower bounds
        :type lb: numpy.array
        :param ub: Upper bounds
        :type ub: numpy.array
        :param int_var: Indices of integer variables. If None, [], or
                        np.array([]) we assume all variables are continuous.
        :type int_var: numpy.array

        :return: Experimental design of size num_pts x dim
        :rtype: numpy.ndarray
        """
        if int_var is None or len(int_var) == 0:
            int_var = np.array([])

        def wrapper():
            return pydoe.lhs(self.dim, self.num_pts, iterations=1)

        return _expdes_dist(wrapper, self.iterations, lb, ub, int_var)
