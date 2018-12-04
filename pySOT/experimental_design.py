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


class LatinHypercube(ExperimentalDesign):
    """Latin Hypercube experimental design.

    :param dim: Number of dimensions
    :type dim: int
    :param num_pts: Number of desired sampling points
    :type num_pts: int
    :param criterion: Sampling criterion

       - "center" or "c"
          Center the points within the sampling intervals
       - "maximin" or "m"
          Maximize the minimum distance between points, but place
          the point in a randomized location within its interval
       - "centermaximin" or "cm"
          Same as "maximin", but centered within the intervals
       - "correlation" or "corr"
          Minimize the maximum correlation coefficient
    :type criterion: string

    :ivar dim: Number of dimensions
    :ivar num_pts: Number of points in the experimental design
    :ivar criterion: Criterion for generating the design
    """
    def __init__(self, dim, num_pts, criterion="c"):
        self.dim = dim
        self.num_pts = num_pts
        self.criterion = criterion

    def generate_points(self):
        """Generate a Latin hypercube design in the unit hypercube.

        :return: Latin hypercube design in unit hypercube of size num_pts x dim
        :rtype: numpy.array
        """
        return pydoe.lhs(self.dim, self.num_pts, self.criterion)


class SymmetricLatinHypercube(ExperimentalDesign):
    """Symmetric Latin hypercube experimental design.

    :param dim: Number of dimensions
    :type dim: int
    :param num_pts: Number of desired sampling points
    :type num_pts: int

    :ivar dim: Number of dimensions
    :ivar num_pts: Number of points in the experimental design
    """
    def __init__(self, dim, num_pts):
        self.dim = dim
        self.num_pts = num_pts

    def generate_points(self):
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

    def generate_points(self):
        """Generate a two factorial design in the unit hypercube.

        :return: Two factorial design in unit hypercube of size num_pts x dim
        :rtype: numpy.array
        """
        return np.array(list(itertools.product([0, 1], repeat=self.dim)))
