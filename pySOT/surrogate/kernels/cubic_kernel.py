from .kernel import Kernel


class CubicKernel(Kernel):
    """Cubic RBF kernel

    This is a class for the Cubic RBF kernel: :math:`\\varphi(r) = r^3` which
    is conditionally positive definite of order 2.
    """

    def __init__(self):
        super().__init__()
        self.order = 2

    def eval(self, dists):
        """Evaluates the Cubic kernel for a distance matrix

        :param dists: Distance input matrix
        :type dists: numpy.array

        :returns: a matrix where element :math:`(i,j)` is
            :math:`\\|x_i - x_j \\|^3`
        :rtype: numpy.array
        """
        return dists ** 3

    def deriv(self, dists):
        """Evaluates the derivative of the Cubic kernel for a distance matrix.

        :param dists: Distance input matrix
        :type dists: numpy.array

        :returns: a matrix where element :math:`(i,j)` is
            :math:`3 \\| x_i - x_j \\|^2`
        :rtype: numpy.array
        """
        return 3 * dists ** 2
