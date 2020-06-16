import numpy as np

from .optimization_problem import OptimizationProblem


class Michalewicz(OptimizationProblem):
    """Michalewicz function

    .. math::
        f(x_1,\\ldots,x_n) = -\\sum_{i=1}^n \\sin(x_i) \\sin^{20}
            \\left( \\frac{ix_i^2}{\\pi} \\right)

    subject to

    .. math::
        0 \\leq x_i \\leq \\pi

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
        self.lb = np.zeros(dim)
        self.ub = np.pi * np.ones(dim)
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional Michalewicz function \n" + "Global optimum: ??"

    def eval(self, x):
        """Evaluate the Michalewicz function  at x.

        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: float
        """
        self.__check_input__(x)
        return -np.sum(np.sin(x) * (np.sin(((1 + np.arange(self.dim)) * x ** 2) / np.pi)) ** 20)
