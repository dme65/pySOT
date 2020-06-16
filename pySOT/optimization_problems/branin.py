import numpy as np

from .optimization_problem import OptimizationProblem


class Branin(OptimizationProblem):
    """Branin function

    Details: http://www.sfu.ca/~ssurjano/branin.html

    Global optimum: :math:`f(-\\pi,12.275)=0.397887`

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
        self.min = 0.397887
        self.minimum = np.array([-np.pi, 12.275])
        self.dim = 2
        self.lb = -3.0 * np.ones(2)
        self.ub = 3.0 * np.ones(2)
        self.int_var = np.array([])
        self.cont_var = np.arange(0, 2)
        self.info = "2-dimensional Branin function \nGlobal optimum: " + "f(-pi, 12.275) = 0.397887"

    def eval(self, x):
        """Evaluate the Branin function at x

        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: float
        """
        self.__check_input__(x)
        x1 = x[0]
        x2 = x[1]

        t = 1 / (8 * np.pi)
        s = 10
        r = 6
        c = 5 / np.pi
        b = 5.1 / (4 * np.pi ** 2)
        a = 1

        term1 = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2
        term2 = s * (1 - t) * np.cos(x1)

        return term1 + term2 + s
