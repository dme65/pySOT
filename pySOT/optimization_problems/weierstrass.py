import numpy as np

from .optimization_problem import OptimizationProblem


class Weierstrass(OptimizationProblem):
    def __init__(self, dim=10):
        self.dim = dim
        self.min = 0
        self.minimum = np.zeros(dim)
        self.lb = -5 * np.ones(dim)
        self.ub = 5 * np.ones(dim)
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional Weierstrass function"

    def eval(self, x):
        """Evaluate the Weierstrass function at x.

        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: float
        """
        self.__check_input__(x)
        d = len(x)
        f0, val = 0.0, 0.0
        for k in range(12):
            f0 += 1.0 / (2 ** k) * np.cos(np.pi * (3 ** k))
            for i in range(d):
                val += 1.0 / (2 ** k) * np.cos(2 * np.pi * (3 ** k) * (x[i] + 0.5))
        return 10 * ((1.0 / float(d) * val - f0) ** 3)
