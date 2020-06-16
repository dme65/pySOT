import numpy as np

from .optimization_problem import OptimizationProblem


class Griewank(OptimizationProblem):
    """Griewank function

    .. math::
        f(x_1,\\ldots,x_n) = 1 + \\frac{1}{4000} \\sum_{j=1}^n x_j^2 - \
        \\prod_{j=1}^n \\cos \\left( \\frac{x_i}{\\sqrt{i}} \\right)

    subject to

    .. math::
        -512 \\leq x_i \\leq 512

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
        self.lb = -512 * np.ones(dim)
        self.ub = 512 * np.ones(dim)
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional Griewank function \n" + "Global optimum: f(0,0,...,0) = 0"

    def eval(self, x):
        """Evaluate the Griewank function at x.

        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: float
        """
        self.__check_input__(x)
        total = 1
        for i, y in enumerate(x):
            total *= np.cos(y / np.sqrt(i + 1))
        return 1.0 / 4000.0 * sum([y ** 2 for y in x]) - total + 1
