"""
.. module:: optimization_problems
  :synopsis: Optimization test problems for multi-modal and
             box-constrained global optimization

.. moduleauthor:: David Eriksson <dme65@cornell.edu>,
                 David Bindel <bindel@cornell.edu>

:Module: optimization_problems
:Author: David Eriksson <dme65@cornell.edu>,
        David Bindel <bindel@cornell.edu>
"""

import numpy as np
import abc
from abc import abstractmethod


class OptimizationProblem(object):
    """Base class for optimization problems."""
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.dim = None
        self.lb = None
        self.ub = None
        self.int_var = None
        self.cont_var = None

    def __check_input__(self, x):
        if len(x) != self.dim:
            raise ValueError('Dimension mismatch')

    @abstractmethod
    def eval(self, record):  # pragma: no cover
        pass

# ========================= 2-dimensional =======================


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
        fact1b = 19 - 14 * x1 + 3 * x1 ** 2 - \
            14 * x2 + 6 * x1 * x2 + 3 * x2 ** 2
        fact1 = 1 + fact1a * fact1b

        fact2a = (2 * x1 - 3 * x2) ** 2
        fact2b = 18 - 32 * x1 + 12 * x1 ** 2 + 48 * x2 - \
            36 * x1 * x2 + 27 * x2 ** 2
        fact2 = 30 + fact2a * fact2b

        return fact1 * fact2


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
        self.info = "2-dimensional Six-hump function \nGlobal optimum: " +\
                    "f(0.0898, -0.7126) = -1.0316"

    def eval(self, x):
        """Evaluate the Six Hump Camel function at x

        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: float
        """
        self.__check_input__(x)
        return (4.0 - 2.1*x[0]**2 + (x[0]**4)/3.0)*x[0]**2 + \
            x[0]*x[1] + (-4 + 4*x[1]**2) * x[1]**2


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
        self.info = "2-dimensional Branin function \nGlobal optimum: " +\
                    "f(-pi, 12.275) = 0.397887"

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

# ========================= 3-dimensional =======================


class Hartman3(OptimizationProblem):
    """Hartman 3 function

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
        self.info = "3-dimensional Hartman function \nGlobal optimum: " +\
                    "f(0.114614,0.555649,0.852547) = -3.86278"

    def eval(self, x):
        """Evaluate the Hartman 3 function at x

        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: float
        """
        self.__check_input__(x)
        alpha = np.array([1, 1.2, 3, 3.2])
        A = np.array([[3.0, 10.0, 30.0], [0.1, 10.0, 35.0],
                     [3.0, 10.0, 30.0], [0.1, 10.0, 35.0]])
        P = np.array([[0.3689, 0.1170, 0.2673],
                     [0.4699, 0.4387, 0.747],
                     [0.1091, 0.8732, 0.5547],
                     [0.0381, 0.5743, 0.8828]])
        outer = 0
        for ii in range(4):
            inner = 0
            for jj in range(3):
                xj = x[jj]
                Aij = A[ii, jj]
                Pij = P[ii, jj]
                inner += Aij * ((xj-Pij) ** 2)
            outer += alpha[ii] * np.exp(-inner)
        return -outer


# =========================6-dimensional =======================


class Hartman6(OptimizationProblem):
    """Hartman 6 function

    Details: http://www.sfu.ca/~ssurjano/hart6.html

    Global optimum: :math:`f(0.201,0.150,0.476,0.275,0.311,0.657)=-3.322`

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
        self.min = -3.32237
        self.minimum = np.array([0.20169, 0.150011, 0.476874,
                                 0.275332, 0.311652, 0.6573])
        self.dim = 6
        self.lb = np.zeros(6)
        self.ub = np.ones(6)
        self.int_var = np.array([])
        self.cont_var = np.arange(0, 6)
        self.info = "6-dimensional Hartman function \nGlobal optimum: " + \
                    "f(0.2016,0.15001,0.47687,0.27533,0.31165,0.657) = -3.3223"

    def eval(self, x):
        """Evaluate the Hartman 6 function at x

        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: float
        """
        self.__check_input__(x)
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array([[10.0, 3.0,  17.0, 3.5,  1.7,  8.0],
                      [0.05, 10.0, 17.0, 0.1,  8.0,  14.0],
                      [3.0,  3.5,  1.7,  10.0, 17.0, 8.0],
                      [17.0, 8.0,  0.05, 10.0, 0.1,  14.0]])
        P = 1e-4 * np.array([[1312.0, 1696.0, 5569.0, 124.0,  8283.0, 5886.0],
                             [2329.0, 4135.0, 8307.0, 3736.0, 1004.0, 9991.0],
                             [2348.0, 1451.0, 3522.0, 2883.0, 3047.0, 6650.0],
                             [4047.0, 8828.0, 8732.0, 5743.0, 1091.0, 381.0]])
        outer = 0
        for ii in range(4):
            inner = 0
            for jj in range(6):
                xj = x[jj]
                Aij = A[ii, jj]
                Pij = P[ii, jj]
                inner += Aij * ((xj - Pij) ** 2)
            outer += alpha[ii] * np.exp(-inner)
        return -outer

# ========================= n-dimensional =======================


class Rastrigin(OptimizationProblem):
    """Rastrigin function

    .. math::
        f(x_1,\\ldots,x_n)=10n-\\sum_{i=1}^n (x_i^2 - 10 \\cos(2 \\pi x_i))

    subject to

    .. math::
        -5.12 \\leq x_i \\leq 5.12

    Global optimum: :math:`f(0,0,...,0)=0`

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
        self.minimum = np.zeros(dim)
        self.lb = -5.12 * np.ones(dim)
        self.ub = 5.12 * np.ones(dim)
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional Rastrigin function \n" + \
                               "Global optimum: f(0,0,...,0) = 0"

    def eval(self, x):
        """Evaluate the Rastrigin function  at x

        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: float
        """
        self.__check_input__(x)
        return 10 * self.dim + sum(x**2 - 10 * np.cos(2 * np.pi * x))


class Ackley(OptimizationProblem):
    """Ackley function

    .. math::
        f(x_1,\\ldots,x_n) = -20\\exp\\left( -0.2 \\sqrt{\\frac{1}{n} \
        \\sum_{j=1}^n x_j^2} \\right) -\\exp \\left( \\frac{1}{n} \
        \\sum{j=1}^n \\cos(2 \\pi x_j) \\right) + 20 - e

    subject to

    .. math::
        -15 \\leq x_i \\leq 20

    Global optimum: :math:`f(0,0,...,0)=0`

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
        self.minimum = np.zeros(dim)
        self.lb = -15 * np.ones(dim)
        self.ub = 20 * np.ones(dim)
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional Ackley function \n" +\
                               "Global optimum: f(0,0,...,0) = 0"

    def eval(self, x):
        """Evaluate the Ackley function  at x

        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: float
        """
        self.__check_input__(x)
        d = float(self.dim)
        return -20.0 * np.exp(-0.2*np.sqrt(np.sum(x**2) / d)) - \
            np.exp(np.sum(np.cos(2.0*np.pi*x)) / d) + 20 + np.exp(1)


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
        self.info = str(dim) + "-dimensional Michalewicz function \n" + \
                               "Global optimum: ??"

    def eval(self, x):
        """Evaluate the Michalewicz function  at x.

        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: float
        """
        self.__check_input__(x)
        return -np.sum(np.sin(x) * (
            np.sin(((1 + np.arange(self.dim)) * x**2)/np.pi)) ** 20)


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
        self.info = str(dim) + "-dimensional Levy function \n" +\
                               "Global optimum: f(1,1,...,1) = 0"

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
        return np.sin(np.pi*w[0]) ** 2 + \
            np.sum((w[1:d-1]-1)**2 * (1 + 10*np.sin(np.pi*w[1:d-1]+1)**2)) + \
            (w[d-1] - 1)**2 * (1 + np.sin(2*np.pi*w[d-1])**2)


class Griewank(OptimizationProblem):
    """Griewank function

    .. math::
        f(x_1,\\ldots,x_n) = 1 + \\frac{1}{4000} \\sum_{j=1}^n x_j^2 - \
        \\prod_{j=1}^n \\cos \\left( \\frac{x_i}{\\sqrt{i}} \\right)

    subject to

    .. math::
        -512 \\leq x_i \\leq 512

    Global optimum: :math:`f(0,0,...,0)=0`

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
        self.minimum = np.zeros(dim)
        self.lb = -512 * np.ones(dim)
        self.ub = 512 * np.ones(dim)
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional Griewank function \n" +\
                               "Global optimum: f(0,0,...,0) = 0"

    def eval(self, x):
        """Evaluate the Griewank function at x.

        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: float
        """
        self.__check_input__(x)
        total = 1
        for i, y in enumerate(x):
            total *= np.cos(y / np.sqrt(i + 1))
        return 1.0 / 4000.0 * sum([y**2 for y in x]) - total + 1


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
        self.info = str(dim) + "-dimensional Rosenbrock function \n" +\
                               "Global optimum: f(1,1,...,1) = 0"

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
            total += 100 * (x[i] ** 2 - x[i+1]) ** 2 + (x[i] - 1) ** 2
        return total


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
        self.info = str(dim) + "-dimensional Schwefel function \n" +\
                               "Global optimum: f(420.9687,...,420.9687) = 0"

    def eval(self, x):
        """Evaluate the Schwefel function at x.

        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: float
        """
        self.__check_input__(x)
        return 418.9829 * self.dim - \
            sum([y * np.sin(np.sqrt(abs(y))) for y in x])


class Sphere(OptimizationProblem):
    """Sphere function

    .. math::
        f(x_1,\\ldots,x_n) = \\sum_{j=1}^n x_j^2

    subject to

    .. math::
        -5.12 \\leq x_i \\leq 5.12

    Global optimum: :math:`f(0,0,...,0)=0`

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
        self.minimum = np.zeros(dim)
        self.lb = -5.12 * np.ones(dim)
        self.ub = 5.12 * np.ones(dim)
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional Sphere function \n" + \
                               "Global optimum: f(0,0,...,0) = 0"

    def eval(self, x):
        """Evaluate the Sphere function at x.

        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: float
        """
        self.__check_input__(x)
        return np.sum(x ** 2)


class Exponential(OptimizationProblem):
    """Exponential function

    .. math::
        f(x_1,\\ldots,x_n) = \\sum_{j=1}^n e^{jx_j} - \\sum_{j=1} e^{-5.12 j}

    subject to

    .. math::
        -5.12 \\leq x_i \\leq 5.12

    Global optimum: :math:`f(0,0,...,0)=0`

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
        self.minimum = -5.12 * np.ones(dim)
        self.lb = -5.12 * np.ones(dim)
        self.ub = 5.12 * np.ones(dim)
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional Exponential function \n" +\
                               "Global optimum: f(-5.12,-5.12,...,-5.12) = 0"

    def eval(self, x):
        """Evaluate the Exponential function  at x.

        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: float
        """
        self.__check_input__(x)
        total = 0.0
        for i in range(len(x)):
            total += np.exp((i+1)*x[i-1]) - np.exp(-5.12*(i+1))
        return total


class Himmelblau(OptimizationProblem):
    """Himmelblau function

    .. math::
        f(x_1,\\ldots,x_n) = 10n -
            \\frac{1}{2n} \\sum_{i=1}^n (x_i^4 - 16x_i^2 + 5x_i)

    Global optimum: :math:`f(-2.903,...,-2.903)=-39.166`

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
        self.min = -39.166165703771412
        self.minimum = -2.903534027771178 * np.ones(dim)
        self.lb = -5.12 * np.ones(dim)
        self.ub = 5.12 * np.ones(dim)
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional Himmelblau function \n" + \
                               "Global optimum: f(-2.903,...,-2.903) = -39.166"

    def eval(self, x):
        """Evaluate the Himmelblau function at x.

        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: float
        """
        self.__check_input__(x)
        return 0.5 * np.sum(x**4 - 16*x**2 + 5*x) / float(self.dim)


class Zakharov(OptimizationProblem):
    """Zakharov function

    Global optimum: :math:`f(0,0,...,0)=1`

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
        self.minimum = np.zeros(dim)
        self.lb = -5 * np.ones(dim)
        self.ub = 10 * np.ones(dim)
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional Zakharov function \n" + \
                               "Global optimum: f(0,0,...,0) = 1"

    def eval(self, x):
        """Evaluate the Zakharov function at x.

        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: float
        """
        self.__check_input__(x)
        return np.sum(x**2) + np.sum(0.5*(1 + np.arange(self.dim))*x)**2 + \
            np.sum(0.5*(1 + np.arange(self.dim))*x)**4


class SumOfSquares(OptimizationProblem):
    """Sum of squares function

    .. math::
        f(x_1,\\ldots,x_n)=\\sum_{i=1}^n ix_i^2

    Global optimum: :math:`f(0,0,...,0)=0`

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
        self.minimum = np.zeros(dim)
        self.lb = -5 * np.ones(dim)
        self.ub = 5 * np.ones(dim)
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional SumOfSquares function \n" + \
                               "Global optimum: f(0,0,...,0) = 0"

    def eval(self, x):
        """Evaluate the Sum of squares function at x.

        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: float
        """
        self.__check_input__(x)
        return np.sum((1 + np.arange(self.dim)) * x**2)


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
        self.info = str(dim) + "-dimensional Perm function \n" + \
                               "Global optimum: f(1,1/2,1/3...,1/d) = 0"

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
                inner += ((jj+1) + beta) * (xj**(ii+1) - (1.0/(jj+1))**(ii+1))
            outer += inner**2
        return outer


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
                val += 1.0 / (2**k) * np.cos(2*np.pi * (3**k) * (x[i] + 0.5))
        return 10 * ((1.0 / float(d) * val - f0) ** 3)
