import numpy as np

from .optimization_problem import OptimizationProblem


class Perm(OptimizationProblem):
    """Perm function

    Global optimum: :math:`f(1,1/2,1/3,...,1/n)=0`

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
        self.minimum = np.ones(dim) / np.arange(1, dim + 1)
        self.lb = -5 * np.ones(dim)
        self.ub = 5 * np.ones(dim)
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional Perm function \n" + "Global optimum: f(1,1/2,1/3...,1/d) = 0"

    def eval(self, x):
        """Evaluate the Perm function at x.

        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: float
        """
        self.__check_input__(x)
        beta = 10.0
        d = len(x)
        outer = 0.0
        for ii in range(d):
            inner = 0.0
            for jj in range(d):
                xj = x[jj]
                inner += ((jj + 1) + beta) * (xj ** (ii + 1) - (1.0 / (jj + 1)) ** (ii + 1))
            outer += inner ** 2
        return outer
