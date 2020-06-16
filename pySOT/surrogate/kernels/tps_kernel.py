import numpy as np

from .kernel import Kernel


class TPSKernel(Kernel):
    """Thin-plate spline RBF kernel.

    This is a basic class for the TPS RBF kernel:
    :math:`\\varphi(r) = r^2 \\log(r)` which is
    conditionally positive definite of order 2.
    """

    def __init__(self):
        super().__init__()
        self.order = 2

    def eval(self, dists):
        """Evaluate the TPS kernel.

        :param dists: Distance input matrix
        :type dists: numpy.array

        :returns: a matrix where element :math:`(i,j)` is
            :math:`\\|x_i - x_j \\|^2 \\log (\\|x_i - x_j \\|)`
        :rtype: numpy.array
        """
        dists[dists < np.finfo(float).eps] = np.finfo(float).eps
        return (dists ** 2) * np.log(dists)

    def deriv(self, dists):
        """Evaluate the derivative of the TPS kernel.

        :param dists: Distance input matrix
        :type dists: numpy.array

        :returns: a matrix where element :math:`(i,j)` is
            :math:`\\|x_i - x_j \\|(1 + 2 \\log (\\|x_i - x_j \\|) )`
        :rtype: numpy.array
        """
        dists[dists < np.finfo(float).eps] = np.finfo(float).eps
        return dists * (1 + 2 * np.log(dists))
