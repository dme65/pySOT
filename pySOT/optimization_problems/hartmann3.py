import numpy as np

from .optimization_problem import OptimizationProblem


class Hartmann3(OptimizationProblem):
    """Hartmann 3 function

    Details: http://www.sfu.ca/~ssurjano/hart3.html

    Global optimum: :math:`f(0.114614,0.555649,0.852547)=-3.86278`

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
        self.dim = 3
        self.lb = np.zeros(3)
        self.ub = np.ones(3)
        self.int_var = np.array([])
        self.cont_var = np.arange(0, 3)
        self.min = -3.86278
        self.minimum = np.array([0.114614, 0.555649, 0.852547])
        self.info = "3-dimensional Hartmann function \nGlobal optimum: " + "f(0.114614,0.555649,0.852547) = -3.86278"

    def eval(self, x):
        """Evaluate the Hartmann 3 function at x

        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: float
        """
        self.__check_input__(x)
        alpha = np.array([1, 1.2, 3, 3.2])
        A = np.array([[3.0, 10.0, 30.0], [0.1, 10.0, 35.0], [3.0, 10.0, 30.0], [0.1, 10.0, 35.0]])
        P = np.array(
            [[0.3689, 0.1170, 0.2673], [0.4699, 0.4387, 0.747], [0.1091, 0.8732, 0.5547], [0.0381, 0.5743, 0.8828]]
        )
        outer = 0
        for ii in range(4):
            inner = 0
            for jj in range(3):
                xj = x[jj]
                Aij = A[ii, jj]
                Pij = P[ii, jj]
                inner += Aij * ((xj - Pij) ** 2)
            outer += alpha[ii] * np.exp(-inner)
        return -outer
