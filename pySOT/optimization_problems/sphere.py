import numpy as np

from .optimization_problem import OptimizationProblem


class Sphere(OptimizationProblem):
    """Sphere function

    .. math::
        f(x_1,\\ldots,x_n) = \\sum_{j=1}^n x_j^2

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
        self.minimum = np.zeros(dim)
        self.lb = -5.12 * np.ones(dim)
        self.ub = 5.12 * np.ones(dim)
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional Sphere function \n" + "Global optimum: f(0,0,...,0) = 0"

    def eval(self, x):
        """Evaluate the Sphere function at x.

        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: float
        """
        self.__check_input__(x)
        return np.sum(x ** 2)
