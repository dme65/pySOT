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
import six


@six.add_metaclass(abc.ABCMeta)
class OptimizationProblem(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def dim(self):  # pragma: no cover
        pass

    @abc.abstractmethod
    def lb(self):  # pragma: no cover
        pass

    @abc.abstractmethod
    def ub(self):  # pragma: no cover
        pass

    @abc.abstractmethod
    def num_expensive_constraints(self):  # pragma: no cover
        pass

    @abc.abstractmethod
    def num_cheap_constraints(self):  # pragma: no cover
        pass

    @abc.abstractmethod
    def eval_cheap_constraints(self, X):  # pragma: no cover
        pass

    @abc.abstractmethod
    def deval_cheap_constraints(self, X):  # pragma: no cover
        pass

    @abc.abstractmethod
    def integer_variables(self):  # pragma: no cover
        pass

    @abc.abstractmethod
    def continuous_variables(self):  # pragma: no cover
        pass

    @abc.abstractmethod
    def eval(self, record):  # pragma: no cover
        return

# ========================= 2-dimensional =======================


class GoldsteinPrice(OptimizationProblem):
    def __init__(self):
        self.info = "2-dimensional Goldstein-Price function"
        self.min = 3.0
        self.minimum = np.array([0, -1])

    def dim(self):
        return 2

    def lb(self):
        return -2.0 * np.ones(2)

    def ub(self):
        return 2.0 * np.ones(2)

    def num_expensive_constraints(self):
        return 0

    def num_cheap_constraints(self):
        return 0

    def eval_cheap_constraints(self, X):
        raise NotImplementedError("There are no cheap constraints")

    def deval_cheap_constraints(self, X):
        raise NotImplementedError("There are no cheap constraints")

    def integer_variables(self):
        return np.array([])

    def continuous_variables(self):
        return np.arange(0, 2)

    def eval(self, xx):
        if len(xx) != 2:
            raise ValueError('Dimension mismatch')

        x1 = xx[0]
        x2 = xx[1]

        fact1a = (x1 + x2 + 1) ** 2
        fact1b = 19 - 14 * x1 + 3 * x1 ** 2 - 14 * x2 + 6 * x1 * x2 + 3 * x2 ** 2
        fact1 = 1 + fact1a * fact1b

        fact2a = (2 * x1 - 3 * x2) ** 2
        fact2b = 18 - 32 * x1 + 12 * x1 ** 2 + 48 * x2 - 36 * x1 * x2 + 27 * x2 ** 2
        fact2 = 30 + fact2a * fact2b

        return fact1 * fact2


class SixHumpCamel(OptimizationProblem):
    def __init__(self):
        self.info = "2-dimensional Six-hump camel function"
        self.min = -1.0316
        self.minimum = np.array([0.0898, -0.7126])

    def dim(self):
        return 2

    def lb(self):
        return -3.0 * np.ones(2)

    def ub(self):
        return 3.0 *np.ones(2)

    def num_expensive_constraints(self):
        return 0

    def num_cheap_constraints(self):
        return 0

    def eval_cheap_constraints(self, X):
        raise NotImplementedError("There are no cheap constraints")

    def deval_cheap_constraints(self, X):
        raise NotImplementedError("There are no cheap constraints")

    def integer_variables(self):
        return np.array([])

    def continuous_variables(self):
        return np.arange(0, 2)

    def eval(self, x):
        if len(x) != 2:
            raise ValueError('Dimension mismatch')
        return (4.0 - 2.1*x[0]**2 + (x[0]**4)/3.0)*x[0]**2 + x[0]*x[1] + (-4 + 4*x[1]**2) * x[1]**2


class Branin(OptimizationProblem):
    def __init__(self):
        self.info = "2-dimensional Branin function"
        self.min = 0.397887
        self.minimum = np.array([-np.pi, 12.275])

    def dim(self):
        return 2

    def lb(self):
        return -3.0 * np.ones(2)

    def ub(self):
        return 3.0 *np.ones(2)

    def num_expensive_constraints(self):
        return 0

    def num_cheap_constraints(self):
        return 0

    def eval_cheap_constraints(self, X):
        raise NotImplementedError("There are no cheap constraints")

    def deval_cheap_constraints(self, X):
        raise NotImplementedError("There are no cheap constraints")

    def integer_variables(self):
        return np.array([])

    def continuous_variables(self):
        return np.arange(0, 2)

    def eval(self, xx):
        if len(xx) != 2:
            raise ValueError('Dimension mismatch')
        x1 = xx[0]
        x2 = xx[1]

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
    """

    def __init__(self):
        self.info = "3-dimensional Hartman function \nGlobal optimum: " +\
                    "f(0.114614,0.555649,0.852547) = -3.86278"
        self.min = -3.86278
        self.minimum = np.array([0.114614, 0.555649, 0.852547])

    def dim(self):
        return 3

    def lb(self):
        return np.zeros(3)

    def ub(self):
        return np.ones(3)

    def num_expensive_constraints(self):
        return 0

    def num_cheap_constraints(self):
        return 0

    def eval_cheap_constraints(self, X):
        raise NotImplementedError("There are no cheap constraints")

    def deval_cheap_constraints(self, X):
        raise NotImplementedError("There are no cheap constraints")

    def integer_variables(self):
        return np.array([])

    def continuous_variables(self):
        return np.arange(0, 3)

    def eval(self, x):
        """Evaluate the Hartman 3 function  at x

        :param x: Data point
        :return: Value at x
        """
        if len(x) != 3:
            raise ValueError('Dimension mismatch')
        alpha = np.array([1, 1.2, 3, 3.2])
        A = np.matrix([[3.0, 10.0, 30.0], [0.1, 10.0, 35.0],
                       [3.0, 10.0, 30.0], [0.1, 10.0, 35.0]])
        P = np.matrix([[0.3689, 0.1170, 0.2673],
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

    Global optimum: :math:`f(0.20169,0.150011,0.476874,0.275332,0.311652,0.6573)=-3.32237`
    """

    def __init__(self):
        self.info = "6-dimensional Hartman function \nGlobal optimum: " + \
                    "f(0.20169,0.150011,0.476874,0.275332,0.311652,0.6573) = -3.32237"
        self.min = -3.32237
        self.minimum = np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])

    def dim(self):
        return 6

    def lb(self):
        return np.zeros(6)

    def ub(self):
        return np.ones(6)

    def num_expensive_constraints(self):
        return 0

    def num_cheap_constraints(self):
        return 0

    def eval_cheap_constraints(self, X):
        raise NotImplementedError("There are no cheap constraints")

    def deval_cheap_constraints(self, X):
        raise NotImplementedError("There are no cheap constraints")

    def integer_variables(self):
        return np.array([])

    def continuous_variables(self):
        return np.arange(0, 6)

    def eval(self, x):
        """Evaluate the Hartman 3 function  at x

        :param x: Data point
        :return: Value at x
        """

        if len(x) != 6:
            raise ValueError('Dimension mismatch')
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.matrix([[10.0, 3.0,  17.0, 3.5,  1.7,  8.0 ],
                       [0.05, 10.0, 17.0, 0.1,  8.0,  14.0],
                       [3.0,  3.5,  1.7,  10.0, 17.0, 8.0 ],
                       [17.0, 8.0,  0.05, 10.0, 0.1,  14.0]])
        P = 1E-4 * np.matrix([[1312.0, 1696.0, 5569.0, 124.0,  8283.0, 5886.0],
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
    """

    def __init__(self, dim=10):
        self.__dim__ = dim
        self.info = str(dim)+"-dimensional Rastrigin function \n" +\
                             "Global optimum: f(0,0,...,0) = 0"
        self.min = 0
        self.minimum = np.zeros(dim)

    def dim(self):
        return self.__dim__

    def lb(self):
        return -5.12 * np.ones(self.__dim__)

    def ub(self):
        return 5.12 * np.ones(self.__dim__)

    def num_expensive_constraints(self):
        return 0

    def num_cheap_constraints(self):
        return 0

    def eval_cheap_constraints(self, X):
        raise NotImplementedError("There are no cheap constraints")

    def deval_cheap_constraints(self, X):
        raise NotImplementedError("There are no cheap constraints")

    def integer_variables(self):
        return np.array([])

    def continuous_variables(self):
        return np.arange(0, self.__dim__)

    def eval(self, x):
        """Evaluate the Rastrigin function  at x

        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: float
        """

        if len(x) != self.__dim__:
            raise ValueError('Dimension mismatch')
        return 10 * self.__dim__ + sum(x**2 - 10 * np.cos(2 * np.pi * x))


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
    """

    def __init__(self, dim=10):
        self.__dim__ = dim
        self.info = str(dim)+"-dimensional Ackley function \n" +\
                             "Global optimum: f(0,0,...,0) = 0"
        self.min = 0
        self.minimum = np.zeros(dim)

    def dim(self):
        return self.__dim__

    def lb(self):
        return -15 * np.ones(self.__dim__)

    def ub(self):
        return 20 * np.ones(self.__dim__)

    def num_expensive_constraints(self):
        return 0

    def num_cheap_constraints(self):
        return 0

    def eval_cheap_constraints(self, X):
        raise NotImplementedError("There are no cheap constraints")

    def deval_cheap_constraints(self, X):
        raise NotImplementedError("There are no cheap constraints")

    def integer_variables(self):
        return np.array([])

    def continuous_variables(self):
        return np.arange(0, self.__dim__)

    def eval(self, x):
        """Evaluate the Ackley function  at x

        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: float
        """

        if len(x) != self.__dim__:
            raise ValueError('Dimension mismatch')
        n = float(self.__dim__)
        return -20.0 * np.exp(-0.2*np.sqrt(np.sum(x**2)/n)) - \
            np.exp(np.sum(np.cos(2.0*np.pi*x))/n) + 20 + np.exp(1)


class Michalewicz(OptimizationProblem):
    """Michalewicz function

    .. math::
        f(x_1,\\ldots,x_n) = -\\sum_{i=1}^n \\sin(x_i) \\sin^{20} \\left( \\frac{ix_i^2}{\\pi} \\right)

    subject to

    .. math::
        0 \\leq x_i \\leq \\pi
    """

    def __init__(self, dim=10):
        self.__dim__ = dim
        self.info = str(dim)+"-dimensional Michalewicz function \n" +\
                             "Global optimum: ??"

    def dim(self):
        return self.__dim__

    def lb(self):
        return np.zeros(self.__dim__)

    def ub(self):
        return np.pi * np.ones(self.__dim__)

    def num_expensive_constraints(self):
        return 0

    def num_cheap_constraints(self):
        return 0

    def eval_cheap_constraints(self, X):
        raise NotImplementedError("There are no cheap constraints")

    def deval_cheap_constraints(self, X):
        raise NotImplementedError("There are no cheap constraints")

    def integer_variables(self):
        return np.array([])

    def continuous_variables(self):
        return np.arange(0, self.__dim__)

    def eval(self, x):
        """Evaluate the Michalewicz function  at x

        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: float
        """

        if len(x) != self.__dim__:
            raise ValueError('Dimension mismatch')
        return -np.sum(np.sin(x) * (np.sin(((1+np.arange(self.__dim__))
                                            * x**2)/np.pi)) ** 20)


class Levy(OptimizationProblem):
    """Levy function

    Details: https://www.sfu.ca/~ssurjano/levy.html

    Global optimum: :math:`f(1,1,...,1)=0`
    """

    def __init__(self, dim=10):
        self.__dim__ = dim
        self.info = str(dim)+"-dimensional Levy function \n" +\
                             "Global optimum: f(1,1,...,1) = 0"
        self.min = 0.0
        self.minimum = np.ones(dim)

    def dim(self):
        return self.__dim__

    def lb(self):
        return -5 * np.ones(self.__dim__)

    def ub(self):
        return 5 * np.ones(self.__dim__)

    def num_expensive_constraints(self):
        return 0

    def num_cheap_constraints(self):
        return 0

    def eval_cheap_constraints(self, X):
        raise NotImplementedError("There are no cheap constraints")

    def deval_cheap_constraints(self, X):
        raise NotImplementedError("There are no cheap constraints")

    def integer_variables(self):
        return np.array([])

    def continuous_variables(self):
        return np.arange(0, self.__dim__)

    def eval(self, x):
        """Evaluate the Levy function  at x

        :param x: Data point
        :return: Value at x
        """
        if len(x) != self.__dim__:
            raise ValueError('Dimension mismatch')
        w = 1 + (x - 1.0) / 4.0
        d = self.__dim__
        return np.sin(np.pi*w[0])**2 + np.sum((w[1:d-1] - 1)**2 * (1 + 10*np.sin(np.pi*w[1:d-1] + 1)**2)) + \
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
    """

    def __init__(self, dim=10):
        self.__dim__ = dim
        self.info = str(dim)+"-dimensional Griewank function \n" +\
                             "Global optimum: f(0,0,...,0) = 0"
        self.min = 0
        self.minimum = np.zeros(dim)

    def dim(self):
        return self.__dim__

    def lb(self):
        return -512 * np.ones(self.__dim__)

    def ub(self):
        return 512 * np.ones(self.__dim__)

    def num_expensive_constraints(self):
        return 0

    def num_cheap_constraints(self):
        return 0

    def eval_cheap_constraints(self, X):
        raise NotImplementedError("There are no cheap constraints")

    def deval_cheap_constraints(self, X):
        raise NotImplementedError("There are no cheap constraints")

    def integer_variables(self):
        return np.array([])

    def continuous_variables(self):
        return np.arange(0, self.__dim__)

    def eval(self, x):
        """Evaluate the Griewank function  at x

        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: float
        """

        if len(x) != self.__dim__:
            raise ValueError('Dimension mismatch')
        total = 1
        for i, y in enumerate(x):
            total *= np.cos(y / np.sqrt(i+1))
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
    """

    def __init__(self, dim=10):
        self.__dim__ = dim
        self.info = str(dim)+"-dimensional Rosenbrock function \n" +\
                             "Global optimum: f(1,1,...,1) = 0"
        self.min = 0
        self.minimum = np.ones(dim)

    def dim(self):
        return self.__dim__

    def lb(self):
        return -2.048 * np.ones(self.__dim__)

    def ub(self):
        return 2.048 * np.ones(self.__dim__)

    def num_expensive_constraints(self):
        return 0

    def num_cheap_constraints(self):
        return 0

    def eval_cheap_constraints(self, X):
        raise NotImplementedError("There are no cheap constraints")

    def deval_cheap_constraints(self, X):
        raise NotImplementedError("There are no cheap constraints")

    def integer_variables(self):
        return np.array([])

    def continuous_variables(self):
        return np.arange(0, self.__dim__)

    def eval(self, x):
        """Evaluate the Rosenbrock function  at x

        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: float
        """

        if len(x) != self.__dim__:
            raise ValueError('Dimension mismatch')
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
    """

    def __init__(self, dim=10):
        self.__dim__ = dim
        self.info = str(dim)+"-dimensional Schwefel function \n" +\
                             "Global optimum: f(420.968746,...,420.968746) = 0"
        self.min = 0
        self.minimum = 420.968746 * np.ones(dim)

    def dim(self):
        return self.__dim__

    def lb(self):
        return -512 * np.ones(self.__dim__)

    def ub(self):
        return 512 * np.ones(self.__dim__)

    def num_expensive_constraints(self):
        return 0

    def num_cheap_constraints(self):
        return 0

    def eval_cheap_constraints(self, X):
        raise NotImplementedError("There are no cheap constraints")

    def deval_cheap_constraints(self, X):
        raise NotImplementedError("There are no cheap constraints")

    def integer_variables(self):
        return np.array([])

    def continuous_variables(self):
        return np.arange(0, self.__dim__)

    def eval(self, x):
        """Evaluate the Schwefel function  at x

        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: float
        """

        if len(x) != self.__dim__:
            raise ValueError('Dimension mismatch')
        return 418.9829 * self.__dim__ - \
            sum([y * np.sin(np.sqrt(abs(y))) for y in x])


class Sphere(OptimizationProblem):
    """Sphere function

    .. math::
        f(x_1,\\ldots,x_n) = \\sum_{j=1}^n x_j^2

    subject to

    .. math::
        -5.12 \\leq x_i \\leq 5.12

    Global optimum: :math:`f(0,0,...,0)=0`
    """

    def __init__(self, dim=10):
        self.__dim__ = dim
        self.info = str(dim)+"-dimensional Sphere function \n" +\
                             "Global optimum: f(0,0,...,0) = 0"
        self.min = 0
        self.minimum = np.zeros(dim)

    def dim(self):
        return self.__dim__

    def lb(self):
        return -5.12 * np.ones(self.__dim__)

    def ub(self):
        return 5.12 * np.ones(self.__dim__)

    def num_expensive_constraints(self):
        return 0

    def num_cheap_constraints(self):
        return 0

    def eval_cheap_constraints(self, X):
        raise NotImplementedError("There are no cheap constraints")

    def deval_cheap_constraints(self, X):
        raise NotImplementedError("There are no cheap constraints")

    def integer_variables(self):
        return np.array([])

    def continuous_variables(self):
        return np.arange(0, self.__dim__)

    def eval(self, x):
        """Evaluate the Sphere function  at x

        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: float
        """

        if len(x) != self.__dim__:
            raise ValueError('Dimension mismatch')
        return np.sum(x ** 2)


class Exponential(OptimizationProblem):
    """Exponential function

    .. math::
        f(x_1,\\ldots,x_n) = \\sum_{j=1}^n e^{jx_j} - \\sum_{j=1} e^{-5.12 j}

    subject to

    .. math::
        -5.12 \\leq x_i \\leq 5.12

    Global optimum: :math:`f(0,0,...,0)=0`
    """

    def __init__(self, dim=10):
        self.__dim__ = dim
        self.info = str(dim)+"-dimensional Exponential function \n" +\
                             "Global optimum: f(-5.12,-5.12,...,-5.12) = 0"
        self.min = 0
        self.minimum = -5.12 * np.ones(dim)

    def dim(self):
        return self.__dim__

    def lb(self):
        return -5.12 * np.ones(self.__dim__)

    def ub(self):
        return 5.12 * np.ones(self.__dim__)

    def num_expensive_constraints(self):
        return 0

    def num_cheap_constraints(self):
        return 0

    def eval_cheap_constraints(self, X):
        raise NotImplementedError("There are no cheap constraints")

    def deval_cheap_constraints(self, X):
        raise NotImplementedError("There are no cheap constraints")

    def integer_variables(self):
        return np.array([])

    def continuous_variables(self):
        return np.arange(0, self.__dim__)

    def eval(self, x):
        """Evaluate the Exponential function  at x

        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: float
        """

        if len(x) != self.__dim__:
            raise ValueError('Dimension mismatch')
        total = 0
        for i in range(len(x)):
            total += np.exp((i+1)*x[i-1]) - np.exp(-5.12*(i+1))
        return total


class Himmelblau(OptimizationProblem):
    def __init__(self, dim=10):
        self.__dim__ = dim
        self.info = str(dim) + "-dimensional Himmelblau function \n" + \
                               "Global optimum: f(-2.903524,-2.903524,...,-2.903524) = -39.166165"
        self.min = -39.166165703771412
        self.minimum = -2.903534027771178 * np.ones(dim)

    def dim(self):
        return self.__dim__

    def lb(self):
        return -5 * np.ones(self.__dim__)

    def ub(self):
        return 5 * np.ones(self.__dim__)

    def num_expensive_constraints(self):
        return 0

    def num_cheap_constraints(self):
        return 0

    def eval_cheap_constraints(self, X):
        raise NotImplementedError("There are no cheap constraints")

    def deval_cheap_constraints(self, X):
        raise NotImplementedError("There are no cheap constraints")

    def integer_variables(self):
        return np.array([])

    def continuous_variables(self):
        return np.arange(0, self.__dim__)

    def eval(self, x):
        if len(x) != self.__dim__:
            raise ValueError('Dimension mismatch')
        return 0.5 * np.sum(x**4 - 16*x**2 + 5*x)/float(self.__dim__)


class Zakharov(OptimizationProblem):
    def __init__(self, dim=10):
        self.__dim__ = dim
        self.info = str(dim) + "-dimensional Zakharov function \n" + \
                               "Global optimum: f(0,0,...,0)=1"
        self.min = 0.0
        self.minimum = np.zeros(dim)

    def dim(self):
        return self.__dim__

    def lb(self):
        return -5 * np.ones(self.__dim__)

    def ub(self):
        return 10 * np.ones(self.__dim__)

    def num_expensive_constraints(self):
        return 0

    def num_cheap_constraints(self):
        return 0

    def eval_cheap_constraints(self, X):
        raise NotImplementedError("There are no cheap constraints")

    def deval_cheap_constraints(self, X):
        raise NotImplementedError("There are no cheap constraints")

    def integer_variables(self):
        return np.array([])

    def continuous_variables(self):
        return np.arange(0, self.__dim__)

    def eval(self, x):
        if len(x) != self.__dim__:
            raise ValueError('Dimension mismatch')
        return np.sum(x**2) + np.sum(0.5*(1+np.arange(self.__dim__))*x)**2 + \
            np.sum(0.5*(1+np.arange(self.__dim__))*x)**4


class SumOfSquares(OptimizationProblem):
    def __init__(self, dim=10):
        self.__dim__ = dim
        self.info = str(dim) + "-dimensional SumOfSquares function \n" + \
                               "Global optimum: f(0,0,...,0)=0"
        self.min = 0.0
        self.minimum = np.zeros(dim)

    def dim(self):
        return self.__dim__

    def lb(self):
        return -5 * np.ones(self.__dim__)

    def ub(self):
        return 5 * np.ones(self.__dim__)

    def num_expensive_constraints(self):
        return 0

    def num_cheap_constraints(self):
        return 0

    def eval_cheap_constraints(self, X):
        raise NotImplementedError("There are no cheap constraints")

    def deval_cheap_constraints(self, X):
        raise NotImplementedError("There are no cheap constraints")

    def integer_variables(self):
        return np.array([])

    def continuous_variables(self):
        return np.arange(0, self.__dim__)

    def eval(self, x):
        if len(x) != self.__dim__:
            raise ValueError('Dimension mismatch')
        return np.sum((1+np.arange(self.__dim__)) * x**2)


class Perm(OptimizationProblem):
    def __init__(self, dim=10):
        self.__dim__ = dim
        self.info = str(dim) + "-dimensional Perm function \n" + \
                               "Global optimum: f(1,1/2,1/3...,1/d)=0"
        self.min = 0.0
        self.minimum = np.ones(dim) / np.arange(1, dim+1)

    def dim(self):
        return self.__dim__

    def lb(self):
        return -5 * np.ones(self.__dim__)

    def ub(self):
        return 5 * np.ones(self.__dim__)

    def num_expensive_constraints(self):
        return 0

    def num_cheap_constraints(self):
        return 0

    def eval_cheap_constraints(self, X):
        raise NotImplementedError("There are no cheap constraints")

    def deval_cheap_constraints(self, X):
        raise NotImplementedError("There are no cheap constraints")

    def integer_variables(self):
        return np.array([])

    def continuous_variables(self):
        return np.arange(0, self.__dim__)

    def eval(self, x):
        if len(x) != self.__dim__:
            raise ValueError('Dimension mismatch')

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


# 20. Weierstrass
class Weierstrass(OptimizationProblem):
    def __init__(self, dim=10):
        self.__dim__ = dim
        self.info = str(dim) + "-dimensional Weierstrass function"
        self.min = 0
        self.minimum = np.zeros(dim)

    def dim(self):
        return self.__dim__

    def lb(self):
        return -5 * np.ones(self.__dim__)

    def ub(self):
        return 5 * np.ones(self.__dim__)

    def num_expensive_constraints(self):
        return 0

    def num_cheap_constraints(self):
        return 0

    def eval_cheap_constraints(self, X):
        raise NotImplementedError("There are no cheap constraints")

    def deval_cheap_constraints(self, X):
        raise NotImplementedError("There are no cheap constraints")

    def integer_variables(self):
        return np.array([])

    def continuous_variables(self):
        return np.arange(0, self.__dim__)

    def eval(self, xx):
        if len(xx) != self.__dim__:
            raise ValueError('Dimension mismatch')
        d = len(xx)
        f0, val = 0.0, 0.0
        for k in range(12):
            f0 += 1.0 / (2 ** k) * np.cos(np.pi * (3 ** k))
            for i in range(d):
                val += 1.0 / (2 ** k) * np.cos(2 * np.pi * (3 ** k) * (xx[i] + 0.5))
        return 10 * ((1.0 / float(d) * val - f0) ** 3)


# ============================== Constraints ==============================


class Keane(OptimizationProblem):
    """Keane's "bump" function

    .. math::
        f(x_1,\\ldots,x_n) = -\\left| \\frac{\\sum_{j=1}^n \\cos^4(x_j) - \
        2 \\prod_{j=1}^n \\cos^2(x_j)}{\\sqrt{\\sum_{j=1}^n jx_j^2}} \\right|

    subject to

    .. math::
        0 \\leq x_i \\leq 5

    .. math::
        0.75 - \\prod_{j=1}^n x_j < 0

    .. math::
        \\sum_{j=1}^n x_j - 7.5n < 0


    Global optimum: -0.835 for large n
    """

    def __init__(self, dim=10):
        self.__dim__ = dim
        self.info = str(dim)+"-dimensional Keane bump function \n" +\
                             "Global optimum: -0.835 for large n"
        self.min = -0.835

    def dim(self):
        return self.__dim__

    def lb(self):
        return np.zeros(self.__dim__)

    def ub(self):
        return 5 * np.ones(self.__dim__)

    def num_expensive_constraints(self):
        return 0

    def num_cheap_constraints(self):
        return 2

    def eval_cheap_constraints(self, X):
        """Evaluate the Keane inequality constraints at x

        :param x: Data points, of size npts x dim
        :type x: numpy.array
        :return: Value at the constraints, of size npts x nconstraints
        :rtype: float
        """

        vec = np.zeros((X.shape[0], 2))
        vec[:, 0] = 0.75 - np.prod(X)
        vec[:, 1] = np.sum(X) - 7.5 * self.__dim__
        return vec

    def deval_cheap_constraints(self, X):
        """Evaluate the derivative of the Keane inequality constraints at x

        :param x: Data points, of size npts x dim
        :type x: numpy.array
        :return: Derivative at the constraints, of size npts x nconstraints x ndims
        :rtype: float
        """

        vec = np.zeros((X.shape[0], 2, X.shape[1]))
        for i in range(X.shape[0]):
            xx = X[i, :]
            for j in range(X.shape[1]):
                vec[i, 0, j] = -np.prod(np.hstack((xx[:j], xx[j + 1:])))
                vec[i, 1, j] = 1.0
        return vec

    def integer_variables(self):
        return np.array([])

    def continuous_variables(self):
        return np.arange(0, self.__dim__)

    def eval(self, x):
        """Evaluate the Keane function at a point x

        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: float
        """

        if len(x) != self.__dim__:
                raise ValueError('Dimension mismatch')
        n = len(x)
        return -abs((sum(np.cos(x)**4)-2 * np.prod(np.cos(x)**2)) /
                    max([1E-10, np.sqrt(np.dot(1+np.arange(n), x**2))]))


class LinearMI(OptimizationProblem):
    """This is a linear mixed integer problem with non-bound constraints

    There are 5 variables, the first 3 are discrete and the last 2
    are continuous.

    Global optimum: :math:`f(1,0,0,0,0) = -1`
    """

    def __init__(self):
        self.info = str(self.dim)+"-dimensional Linear MI \n" +\
                                  "Global optimum: f(1,0,0,0,0) = -1\n" +\
                                  "3 integer variables"
        self.min = -1
        self.minimum = np.array([1, 0, 0, 0, 0])

    def dim(self):
        return 5

    def lb(self):
        return np.zeros(5)

    def ub(self):
        return np.array([10, 10, 10, 1, 1])

    def num_expensive_constraints(self):
        return 0

    def num_cheap_constraints(self):
        return 3

    def eval_cheap_constraints(self, X):
        """Evaluate the inequality constraints at X

        :param X: Data points, of size npts x dim
        :type X: numpy.array
        :return: Value at the constraints, of size npts x nconstraints
        :rtype: float
        """

        vec = np.zeros((X.shape[0], 3))
        vec[:, 0] = X[:, 0] + X[:, 2] - 1.6
        vec[:, 1] = 1.333 * X[:, 1] + X[:, 3] - 3
        vec[:, 2] = - X[:, 2] - X[:, 3] + X[:, 4]
        return vec

    def deval_cheap_constraints(self, X):
        raise NotImplementedError("Not implemented yet")

    def integer_variables(self):
        return np.arange(0, 3)

    def continuous_variables(self):
        return np.arange(3, 5)

    def eval(self, x):
        """Evaluate the LinearMI function  at x

        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: float
        """

        if len(x) != 5:
            raise ValueError('Dimension mismatch')
        return - x[0] + 3 * x[1] + 1.5 * x[2] + 2 * x[3] - 0.5 * x[4]