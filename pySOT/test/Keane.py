import numpy as np


class Keane:
    def __init__(self, dim=10):
        self.xlow = np.zeros(dim)
        self.xup = 5 * np.ones(dim)
        self.dim = dim
        self.min = -0.835
        self.integer = []
        self.continuous = np.arange(0, dim)
        self.info = str(dim)+"-dimensional Keane bump function \n" +\
                             "Global optimum: -0.835 for large n"

    # Return a list with the constraint function at the given point
    def eval_ineq_constraints(self, x):
        vec = np.zeros((x.shape[0], 2))
        vec[:, 0] = 0.75 - np.prod(x)
        vec[:, 1] = np.sum(x) - 7.5 * self.dim
        return vec

    def deriv_ineq_constraints(self, x):
        vec = np.zeros((x.shape[0], 2, x.shape[1]))
        for i in range(x.shape[0]):
            xx = x[i, :]
            for j in range(x.shape[1]):
                vec[i, 0, j] = -np.prod(np.hstack((xx[:j], xx[j+1:])))
                vec[i, 1, j] = 1.0
        return vec

    # Evaluate the objective function for a single data point
    def objfunction(self, x):
        """Evaluate the Keane function  at x

        :param x: Data point
        :return: Value at x
        """
        n = len(x)
        return -abs((sum(np.cos(x)**4)-2 * np.prod(np.cos(x)**2)) /
                    max([1E-10, np.sqrt(np.dot(1+np.arange(n), x**2))]))
