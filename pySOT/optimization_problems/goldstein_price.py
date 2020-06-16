import numpy as np

from .optimization_problem import OptimizationProblem


class GoldsteinPrice(OptimizationProblem):
    def __init__(self):
        self.info = "2-dimensional Goldstein-Price function"
        self.min = 3.0
        self.minimum = np.array([0, -1])
        self.dim = 2
        self.lb = -2.0 * np.ones(2)
        self.ub = 2.0 * np.ones(2)
        self.int_var = np.array([])
        self.cont_var = np.arange(0, 2)

    def eval(self, x):
        """Evaluate the GoldStein Price function at x

        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: float
        """
        self.__check_input__(x)

        x1 = x[0]
        x2 = x[1]

        fact1a = (x1 + x2 + 1) ** 2
        fact1b = 19 - 14 * x1 + 3 * x1 ** 2 - 14 * x2 + 6 * x1 * x2 + 3 * x2 ** 2
        fact1 = 1 + fact1a * fact1b

        fact2a = (2 * x1 - 3 * x2) ** 2
        fact2b = 18 - 32 * x1 + 12 * x1 ** 2 + 48 * x2 - 36 * x1 * x2 + 27 * x2 ** 2
        fact2 = 30 + fact2a * fact2b

        return fact1 * fact2
