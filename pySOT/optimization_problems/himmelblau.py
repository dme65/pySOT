import numpy as np

from .optimization_problem import OptimizationProblem


class Himmelblau(OptimizationProblem):
    """Himmelblau function

    .. math::
        f(x_1,\\ldots,x_n) = 10n -
            \\frac{1}{2n} \\sum_{i=1}^n (x_i^4 - 16x_i^2 + 5x_i)

    Global optimum: :math:`f(-2.903,...,-2.903)=-39.166`

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
        self.min = -39.166165703771412
        self.minimum = -2.903534027771178 * np.ones(dim)
        self.lb = -5.12 * np.ones(dim)
        self.ub = 5.12 * np.ones(dim)
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional Himmelblau function \n" + "Global optimum: f(-2.903,...,-2.903) = -39.166"

    def eval(self, x):
        """Evaluate the Himmelblau function at x.

        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: float
        """
        self.__check_input__(x)
        return 0.5 * np.sum(x ** 4 - 16 * x ** 2 + 5 * x) / float(self.dim)
