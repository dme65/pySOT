"""
.. module:: tails
   :synopsis: Polynomial tails

.. moduleauthor:: David Eriksson <dme65@cornell.edu>,

:Module: tails
:Author: David Eriksson <dme65@cornell.edu>,

"""

import numpy as np


class LinearTail(object):
    """Linear polynomial tail

    This is a standard linear polynomial in d-dimension, built from the basis
    :math:`\{1,x_1,x_2,\ldots,x_d\}`.
    """

    def degree(self):
        """returns the degree of the linear polynomial tail

        :returns: 1
        :rtype: int
        """

        return 1

    def dim_tail(self, dim):
        """returns the dimensionality of the linear polynomial space for a given dimension

        :param dim: Number of dimensions of the Cartesian space
        :type dim: int
        :returns: 1 + dim
        :rtype: int
        """

        return 1 + dim

    def eval(self, X):
        """evaluates the linear polynomial tail for a set of points

        :param X: Points to evaluate, of size npts x dim
        :type X: numpy.array
        :returns: A numpy.array of size npts x dim_tail(dim)
        :rtype: numpy.array
        """

        if len(X.shape) == 1:
            X = np.atleast_2d(X)
        return np.hstack((np.ones((X.shape[0], 1)), X))

    def deriv(self, x):
        """evaluates the gradient of the linear polynomial tail for one point

        :param x: Point to evaluate, of length dim
        :type x: numpy.array
        :returns: A numpy.array of size dim x dim_tail(dim)
        :rtype: numpy.array
        """

        return np.hstack((np.zeros((len(x), 1)), np.eye((len(x)))))


class ConstantTail(object):
    """Constant polynomial tail

    This is a standard linear polynomial in d-dimension, built from the basis
    :math:`\{1\}`.
    """

    def degree(self):
        """returns the degree of the constant polynomial tail

        :returns: 0
        :rtype: int
        """

        return 0

    def dim_tail(self, dim):
        """returns the dimensionality of the constant polynomial space for a given dimension

        :param dim: Number of dimensions of the Cartesian space
        :type dim: int
        :returns: 1
        :rtype: int
        """

        return 1

    def eval(self, X):
        """evaluates the constant polynomial tail for a set of points

        :param X: Points to evaluate, of size npts x dim
        :type X: numpy.array
        :returns: A numpy.array of size npts x dim_tail(dim)
        :rtype: numpy.array
        """

        if len(X.shape) == 1:
            X = np.atleast_2d(X)
        return np.ones((X.shape[0], 1))

    def deriv(self, x):
        """evaluates the gradient of the linear polynomial tail for one point

        :param x: Point to evaluate, of length dim
        :type x: numpy.array
        :returns: A numpy.array of size dim x dim_tail(dim)
        :rtype: numpy.array
        """

        return np.ones((len(x), 1))
