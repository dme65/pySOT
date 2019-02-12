"""
.. module:: experimental_design
  :synopsis: Methods for generating an experimental design.

.. moduleauthor:: David Eriksson <dme65@cornell.edu>,
                 Yi Shen <ys623@cornell.edu>

:Module: experimental_design
:Author: David Eriksson <dme65@cornell.edu>
        Yi Shen <ys623@cornell.edu>
"""

import numpy as np
import pyDOE2 as pydoe
import abc
import six
import itertools
from pySOT.utils import from_unit_box, round_vars
from numpy.linalg import matrix_rank as rank
from scipy.spatial.distance import cdist


@six.add_metaclass(abc.ABCMeta)
class ExperimentalDesign(object):
    """Base class for experimental designs.

    :ivar dim: Number of dimensions
    :ivar num_pts: Number of points in the experimental design
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):  # pragma: no cover
        self.dim = None
        self.num_pts = None

    @abc.abstractmethod
    def generate_points(self):  # pragma: no cover
        pass


def _expdes_dist(gen, iterations, lb, ub, int_var):
    X = None
    best_score = 0
    for _ in range(iterations):
        cand = gen()  # Generate a new design
        if all([x is not None for x in [lb, ub, int_var]]):  # Map and round
            cand = round_vars(from_unit_box(cand, lb, ub), int_var, lb, ub)

        dists = cdist(cand, cand)
        np.fill_diagonal(dists, np.inf)  # Since these are zero
        score = dists.min().min()

        if score > best_score and rank(cand) == cand.shape[1]:
            best_score = score
            X = cand.copy()

    if X is None:
        raise ValueError("No valid design found, increase num_pts?")
    return X


class LatinHypercube(ExperimentalDesign):
    """Latin Hypercube experimental design.

    :param dim: Number of dimensions
    :type dim: int
    :param num_pts: Number of desired sampling points
    :type num_pts: int

    :ivar dim: Number of dimensions
    :ivar num_pts: Number of points in the experimental design
    """
    def __init__(self, dim, num_pts, iterations=100):
        self.dim = dim
        self.num_pts = num_pts
        self.iterations = iterations

    def generate_points(self, lb=None, ub=None, int_var=None):
        """Generate a Latin hypercube design in the unit hypercube.

        :return: Latin hypercube design in unit hypercube of size num_pts x dim
        :rtype: numpy.array
        """
        def wrapper():
            return pydoe.lhs(self.dim, self.num_pts, iterations=1)
        return _expdes_dist(wrapper, self.iterations, lb, ub, int_var)


class SymmetricLatinHypercube(ExperimentalDesign):
    """Symmetric Latin hypercube experimental design.

    :param dim: Number of dimensions
    :type dim: int
    :param num_pts: Number of desired sampling points
    :type num_pts: int

    :ivar dim: Number of dimensions
    :ivar num_pts: Number of points in the experimental design
    """
    def __init__(self, dim, num_pts, iterations=100):
        self.dim = dim
        self.num_pts = num_pts
        self.iterations = iterations

    def generate_points(self, lb=None, ub=None, int_var=None):
        def wrapper():
            return self._slhd()
        return _expdes_dist(wrapper, self.iterations, lb, ub, int_var)

    def _slhd(self):
        """Generate a symmetric Latin hypercube design in the unit hypercube.

        :return: Symmetric Latin hypercube design in the unit hypercube
            of size num_pts x dim
        :rtype: numpy.array
        """
        # Generate a one-dimensional array based on sample number
        points = np.zeros([self.num_pts, self.dim])
        points[:, 0] = np.arange(1, self.num_pts+1)

        # Get the last index of the row in the top half of the hypercube
        middleind = self.num_pts // 2

        # special manipulation if odd number of rows
        if self.num_pts % 2 == 1:
            points[middleind, :] = middleind + 1

        # Generate the top half of the hypercube matrix
        for j in range(1, self.dim):
            for i in range(middleind):
                if np.random.random() < 0.5:
                    points[i, j] = self.num_pts - i
                else:
                    points[i, j] = i + 1
            np.random.shuffle(points[:middleind, j])

        # Generate the bottom half of the hypercube matrix
        for i in range(middleind, self.num_pts):
            points[i, :] = self.num_pts + 1 - points[self.num_pts - 1 - i, :]

        return points/self.num_pts


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

        :return: Two factorial design in unit hypercube of size num_pts x dim
        :rtype: numpy.array
        """
        X = np.array(list(itertools.product([0, 1], repeat=self.dim)))
        if all([x is not None for x in [lb, ub, int_var]]):  # Map and round
            X = round_vars(from_unit_box(X, lb, ub), int_var, lb, ub)
        return X
