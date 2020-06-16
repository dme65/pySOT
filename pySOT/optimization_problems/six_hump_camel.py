import numpy as np

from .optimization_problem import OptimizationProblem


class SixHumpCamel(OptimizationProblem):
    """Six-hump camel function

    Details: https://www.sfu.ca/~ssurjano/camel6.html

    Global optimum: :math:`f(0.0898,-0.7126)=-1.0316`

    :ivar dim: Number of dimensions
    :ivar lb: Lower variable bounds
    :ivar ub: Upper variable bounds
    :ivar int_var: Integer variables
    :ivar cont_var: Continuous variables
    :ivar min: Global minimum value
    :ivar minimum: Global minimizer
    :ivar info: String with problem info
    """

    def __init__(self):
        self.min = -1.0316
        self.minimum = np.array([0.0898, -0.7126])
        self.dim = 2
        self.lb = -3.0 * np.ones(2)
        self.ub = 3.0 * np.ones(2)
        self.int_var = np.array([])
        self.cont_var = np.arange(0, 2)
        self.info = "2-dimensional Six-hump function \nGlobal optimum: " + "f(0.0898, -0.7126) = -1.0316"

    def eval(self, x):
        """Evaluate the Six Hump Camel function at x

        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: float
        """
        self.__check_input__(x)
        return (4.0 - 2.1 * x[0] ** 2 + (x[0] ** 4) / 3.0) * x[0] ** 2 + x[0] * x[1] + (-4 + 4 * x[1] ** 2) * x[1] ** 2
