import numpy as np

from .optimization_problem import OptimizationProblem


class Rosenbrock(OptimizationProblem):
    """Rosenbrock function

    .. math::
        f(x_1,\\ldots,x_n) = \\sum_{j=1}^{n-1} \
        \\left( 100(x_j^2-x_{j+1})^2 + (1-x_j)^2 \\right)

    subject to

    .. math::
        -2.048 \\leq x_i \\leq 2.048

    Global optimum: :math:`f(1,1,...,1)=0`

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
        self.minimum = np.ones(dim)
        self.lb = -2.048 * np.ones(dim)
        self.ub = 2.048 * np.ones(dim)
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional Rosenbrock function \n" + "Global optimum: f(1,1,...,1) = 0"

    def eval(self, x):
        """Evaluate the Rosenbrock function at x

        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: float
        """
        self.__check_input__(x)
        total = 0
        for i in range(len(x) - 1):
            total += 100 * (x[i] ** 2 - x[i + 1]) ** 2 + (x[i] - 1) ** 2
        return total
