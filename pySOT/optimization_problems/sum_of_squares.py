import numpy as np

from .optimization_problem import OptimizationProblem


class SumOfSquares(OptimizationProblem):
    """Sum of squares function

    .. math::
        f(x_1,\\ldots,x_n)=\\sum_{i=1}^n ix_i^2

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
        self.min = 0.0
        self.minimum = np.zeros(dim)
        self.lb = -5 * np.ones(dim)
        self.ub = 5 * np.ones(dim)
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional SumOfSquares function \n" + "Global optimum: f(0,0,...,0) = 0"

    def eval(self, x):
        """Evaluate the Sum of squares function at x.

        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: float
        """
        self.__check_input__(x)
        return np.sum((1 + np.arange(self.dim)) * x ** 2)
