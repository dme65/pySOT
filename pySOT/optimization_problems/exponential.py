import numpy as np

from .optimization_problem import OptimizationProblem


class Exponential(OptimizationProblem):
    """Exponential function

    .. math::
        f(x_1,\\ldots,x_n) = \\sum_{j=1}^n e^{jx_j} - \\sum_{j=1} e^{-5.12 j}

    subject to

    .. math::
        -5.12 \\leq x_i \\leq 5.12

    Global optimum: :math:`f(0,0,...,0)=0`

    :ivar dim: Number of dimensions
    :ivar lb: Lower variable bounds
    :ivar ub: Upper variable bounds
    :ivar int_var: Integer variables
    :ivar cont_var: Continuous variables
    :ivar min: Global minimum value
    :ivar minimum: Global minimizer
    :ivar info: String with problem info
    """

    def __init__(self, dim=10):
        self.dim = dim
        self.min = 0
        self.minimum = -5.12 * np.ones(dim)
        self.lb = -5.12 * np.ones(dim)
        self.ub = 5.12 * np.ones(dim)
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional Exponential function \n" + "Global optimum: f(-5.12,-5.12,...,-5.12) = 0"

    def eval(self, x):
        """Evaluate the Exponential function  at x.

        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: float
        """
        self.__check_input__(x)
        total = 0.0
        for i in range(len(x)):
            total += np.exp((i + 1) * x[i - 1]) - np.exp(-5.12 * (i + 1))
        return total
