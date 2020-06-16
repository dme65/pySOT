import itertools

import numpy as np

from ..utils import from_unit_box, round_vars
from .experimental_design import ExperimentalDesign


class TwoFactorial(ExperimentalDesign):
    """Two-factorial experimental design.

    The two-factorial experimental design consists of the corners
    of the unit hypercube, and hence :math:`2^{dim}` points.

    :param dim: Number of dimensions
    :type dim: int

    :ivar dim: Number of dimensions
    :ivar num_pts: Number of points in the experimental design

    :raises ValueError: If dim >= 15
    """

    def __init__(self, dim):
        if dim >= 15:
            raise ValueError("Refusing to use >= 2^15 points.")
        self.dim = dim
        self.num_pts = 2 ** dim

    def generate_points(self, lb=None, ub=None, int_var=None):
        """Generate a two factorial design in the unit hypercube.

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

        :return: Two factorial design in unit hypercube of size num_pts x dim
        :rtype: numpy.array
        """
        if int_var is None or len(int_var) == 0:
            int_var = np.array([])

        X = np.array(list(itertools.product([0, 1], repeat=self.dim)))
        if all([x is not None for x in [lb, ub]]):  # Map and round
            X = round_vars(from_unit_box(X, lb, ub), int_var, lb, ub)
        return X
