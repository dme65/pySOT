import numpy as np

from .optimization_problem import OptimizationProblem


class Levy(OptimizationProblem):
    """Levy function

    Details: https://www.sfu.ca/~ssurjano/levy.html

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
        self.min = 0.0
        self.minimum = np.ones(dim)
        self.lb = -5 * np.ones(dim)
        self.ub = 5 * np.ones(dim)
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional Levy function \n" + "Global optimum: f(1,1,...,1) = 0"

    def eval(self, x):
        """Evaluate the Levy function at x.

        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: float
        """
        self.__check_input__(x)
        w = 1 + (x - 1.0) / 4.0
        d = self.dim
        return (
            np.sin(np.pi * w[0]) ** 2
            + np.sum((w[1 : d - 1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[1 : d - 1] + 1) ** 2))
            + (w[d - 1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[d - 1]) ** 2)
        )
