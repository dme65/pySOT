import numpy as np

from .kernel import Kernel


class LinearKernel(Kernel):
    """Linear RBF kernel.

     This is a basic class for the Linear RBF kernel:
     :math:`\\varphi(r) = r` which is
     conditionally positive definite of order 1.
     """

    def __init__(self):
        super().__init__()
        self.order = 1

    def eval(self, dists):
        """Evaluate the Linear kernel.

        :param dists: Distance input matrix
        :type dists: numpy.array

        :returns: a matrix where element :math:`(i,j)` is
            :math:`\\|x_i - x_j \\|`
        :rtype: numpy.array
        """
        return dists

    def deriv(self, dists):
        """Evaluate the derivative of the Linear kernel.

        :param dists: Distance input matrix
        :type dists: numpy.array

        :returns: a matrix where element :math:`(i,j)` is 1
        :rtype: numpy.array
        """
        return np.ones(dists.shape)
