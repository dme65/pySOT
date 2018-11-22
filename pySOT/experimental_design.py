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
    __metaclass__ = abc.ABCMeta

    @property
    @abc.abstractmethod
    def dim(self):  # pragma: no cover
        pass

    @property
    @abc.abstractmethod
    def npts(self):  # pragma: no cover
        pass

    @abc.abstractmethod
    def generate_points(self):  # pragma: no cover
        pass


class LatinHypercube(ExperimentalDesign):
    """Latin Hypercube experimental design

    :param dim: Number of dimensions
    :type dim: int
    :param npts: Number of desired sampling points
    :type npts: int
    :param criterion: Sampling criterion
        - "center" or "c"
            center the points within the sampling intervals
        - "maximin" or "m"
            maximize the minimum distance between points, but place the point in a randomized
            location within its interval
        - "centermaximin" or "cm"
            same as "maximin", but
            centered within the intervals
        - "correlation" or "corr"
            minimize the maximum correlation coefficient
    :type criterion: string
    """

    def __init__(self, dim, npts, criterion='c'):
        self._dim = dim
        self._npts = npts
        self.criterion = criterion

    @property
    def dim(self):
        return self._dim

    @property
    def npts(self):
        return self._npts

    def generate_points(self):
        """Generate a matrix with the initial sample points,
        scaled to the unit hypercube

        :return: Latin hypercube design in  the unit cube of size npts x dim
        :rtype: numpy.array
        """

        return pydoe.lhs(self.dim, self.npts, self.criterion)


class SymmetricLatinHypercube(ExperimentalDesign):
    """Symmetric Latin Hypercube experimental design

    :param dim: Number of dimensions
    :type dim: int
    :param npts: Number of desired sampling points
    :type npts: int

    :raises ValueError: If npts < 2*dim
    """

    def __init__(self, dim, npts):
        self._dim = dim
        self._npts = npts
        if npts < 2*dim:
            raise ValueError("We need npts >= 2*dim to make sure the SLHD has full rank")

    @property
    def dim(self):
        return self._dim

    @property
    def npts(self):
        return self._npts

    def _slhd(self):
        """Generate a matrix with the initial sample points,
        scaled to the unit hypercube

        :return: Symmetric Latin hypercube design in the unit cube of size npts x dim
        :rtype: numpy.array
        """

        # Generate a one-dimensional array based on sample number
        points = np.zeros([self.npts, self.dim])
        points[:, 0] = np.arange(1, self.npts+1)

        # Get the last index of the row in the top half of the hypercube
        middleind = self.npts//2

        # special manipulation if odd number of rows
        if self.npts % 2 == 1:
            points[middleind, :] = middleind + 1

        # Generate the top half of the hypercube matrix
        for j in range(1, self.dim):
            for i in range(middleind):
                if np.random.random() < 0.5:
                    points[i, j] = self.npts-i
                else:
                    points[i, j] = i + 1
            np.random.shuffle(points[:middleind, j])

        # Generate the bottom half of the hypercube matrix
        for i in range(middleind, self.npts):
            points[i, :] = self.npts + 1 - points[self.npts - 1 - i, :]

        return points/self.npts

    def generate_points(self):
        """Generate a matrix with the initial sample points,
        scaled to the unit hypercube

        :return: Symmetric Latin hypercube design in the unit cube of size npts x dim
        :rtype: numpy.array
        """

        return self._slhd()


class TwoFactorial(ExperimentalDesign):
    """Two-factorial experimental design

    The two-factorial experimental design consists of the corners
    of the unit hypercube, and hence :math:`2^{dim}` points.

    :param dim: Number of dimensions
    :type dim: int
    :raises ValueError: If dim >= 15
    """

    def __init__(self, dim):
        if dim >= 15:
            raise ValueError("Not generating a design with 2^15 points or more, sorry.")
        self._dim = dim
        self._npts = 2 ** dim

    @property
    def dim(self):
        return self._dim

    @property
    def npts(self):
        return self._npts

    def generate_points(self):
        """Generate a matrix with the initial sample points,
        scaled to the unit hypercube

        :return: Full-factorial design in the unit cube of size (2^dim) x dim
        :rtype: numpy.array
        """

        return np.array(list(itertools.product([0, 1], repeat=self.dim)))
