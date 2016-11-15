"""
.. module:: kernels
   :synopsis: Radial basis function kernels

.. moduleauthor:: David Eriksson <dme65@cornell.edu>,

:Module: kernels
:Author: David Eriksson <dme65@cornell.edu>,

"""

import numpy as np


class CubicKernel(object):
    """Cubic RBF kernel

    This is a basic class for the Cubic RBF kernel: :math:`\\varphi(r) = r^3` which is
    conditionally positive definite of order 2.
    """

    def order(self):
        """returns the order of the Cubic RBF kernel

        :returns: 2
        :rtype: int
        """

        return 2

    def phi_zero(self):
        """returns the value of :math:`\\varphi(0)` for Cubic RBF kernel

        :returns: 0
        :rtype: float
        """

        return 0.0

    def eval(self, dists):
        """evaluates the Cubic kernel for a distance matrix

        :param dists: Distance input matrix
        :type dists: numpy.array
        :returns: a matrix where element :math:`(i,j)` is :math:`\|x_i - x_j \|^3`
        :rtype: numpy.array
        """

        return np.multiply(dists, np.multiply(dists, dists))

    def deriv(self, dists):
        """evaluates the derivative of the Cubic kernel for a distance matrix

        :param dists: Distance input matrix
        :type dists: numpy.array
        :returns: a matrix where element :math:`(i,j)` is :math:`3 \| x_i - x_j \|^2`
        :rtype: numpy.array
        """

        return 3 * np.multiply(dists, dists)


class TPSKernel(object):
    """Thin-plate spline RBF kernel

    This is a basic class for the TPS RBF kernel: :math:`\\varphi(r) = r^2 \log(r)` which is
    conditionally positive definite of order 2.
    """

    def order(self):
        """returns the order of the TPS RBF kernel

        :returns: 2
        :rtype: int
        """

        return 2

    def phi_zero(self):
        """returns the value of :math:`\\varphi(0)` for TPS RBF kernel

        :returns: 0
        :rtype: float
        """

        return 0.0

    def eval(self, dists):
        """evaluates the Cubic kernel for a distance matrix

        :param dists: Distance input matrix
        :type dists: numpy.array
        :returns: a matrix where element :math:`(i,j)` is :math:`\|x_i - x_j \|^2 \log (\|x_i - x_j \|)`
        :rtype: numpy.array
        """

        return np.multiply(np.multiply(dists, dists), np.log(dists + np.finfo(float).tiny))

    def deriv(self, dists):
        """evaluates the derivative of the Cubic kernel for a distance matrix

        :param dists: Distance input matrix
        :type dists: numpy.array
        :returns: a matrix where element :math:`(i,j)` is :math:`\|x_i - x_j \|(1 + 2 \log (\|x_i - x_j \|) )`
        :rtype: numpy.array
        """

        return np.multiply(dists, 1 + 2 * np.log(dists + np.finfo(float).tiny))


class LinearKernel(object):
    """Linear RBF kernel

     This is a basic class for the Linear RBF kernel: :math:`\\varphi(r) = r` which is
     conditionally positive definite of order 1.
     """

    def order(self):
        """returns the order of the Linear RBF kernel

        :returns: 1
        :rtype: int
        """

        return 1

    def phi_zero(self):
        """returns the value of :math:`\\varphi(0)` for Linear RBF kernel

        :returns: 0
        :rtype: float
        """

        return 0

    def eval(self, dists):
        """evaluates the Linear kernel for a distance matrix

        :param dists: Distance input matrix
        :type dists: numpy.array
        :returns: a matrix where element :math:`(i,j)` is :math:`\|x_i - x_j \|`
        :rtype: numpy.array
        """

        return dists

    def deriv(self, dists):
        """evaluates the derivative of the Cubic kernel for a distance matrix

        :param dists: Distance input matrix
        :type dists: numpy.array
        :returns: a matrix where element :math:`(i,j)` is 1
        :rtype: numpy.array
        """

        return np.ones((dists.shape[0], dists.shape[1]))
