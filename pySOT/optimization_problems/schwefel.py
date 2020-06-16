import numpy as np

from .optimization_problem import OptimizationProblem


class Schwefel(OptimizationProblem):
    """Schwefel function

    .. math::
        f(x_1,\\ldots,x_n) = \\sum_{j=1}^{n} \
        \\left( -x_j \\sin(\\sqrt{|x_j|}) \\right) + 418.982997 n

    subject to

    .. math::
        -512 \\leq x_i \\leq 512

    Global optimum: :math:`f(420.968746,420.968746,...,420.968746)=0`

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
        self.minimum = 420.968746 * np.ones(dim)
        self.lb = -512 * np.ones(dim)
        self.ub = 512 * np.ones(dim)
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional Schwefel function \n" + "Global optimum: f(420.9687,...,420.9687) = 0"

    def eval(self, x):
        """Evaluate the Schwefel function at x.

        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: float
        """
        self.__check_input__(x)
        return 418.9829 * self.dim - sum([y * np.sin(np.sqrt(abs(y))) for y in x])
