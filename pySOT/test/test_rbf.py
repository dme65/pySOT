"""
.. module:: test_rbf_regression
  :synopsis: Test RBF regression
.. moduleauthor:: David Eriksson <dme65@cornell.edu>
"""

from pySOT import *
import numpy as np


def main():
    data = Sphere(dim=1)
    maxeval = 100

    kernel = CubicKernel
    tail = LinearTail

    for i in range(2):

        rbf1 = RBFInterpolant(kernel=kernel, tail=tail, maxp=maxeval, dim=data.dim, eta=0)
        rbf2 = RBFInterpolant(kernel=kernel, tail=tail, maxp=maxeval, dim=data.dim, eta='adapt')

        X = np.random.uniform(data.xlow, data.xup, (maxeval, data.dim))
        fX = np.zeros((maxeval, data.dim))
        for j in range(maxeval):
            fX[j] = data.objfunction(X[j, :])
            if i == 0:
                fX[j] += 3*np.cos(10000*X[j, :])
            rbf1.add_point(X[j, :], fX[j])
            rbf2.add_point(X[j, :], fX[j])

        Xpred = np.atleast_2d(np.linspace(data.xlow, data.xup, 1000)).transpose()
        fXpred = np.zeros((1000, data.dim))
        for j in range(1000):
            fXpred[j] = data.objfunction(Xpred[j, :])

        if i == 0:
            print("\nL2 error interpolation with noise: {0}".format(np.linalg.norm(rbf1.evals(Xpred) - fXpred)))
            print("L2 error regularization with noise: {0}".format(np.linalg.norm(rbf2.evals(Xpred) - fXpred)))
        else:
            print("L2 error interpolation without noise: {0}".format(np.linalg.norm(rbf1.evals(Xpred) - fXpred)))
            print("L2 error regularization without noise: {0}\n".format(np.linalg.norm(rbf2.evals(Xpred) - fXpred)))

        """
        import matplotlib.pyplot as plt
        Xpred = np.atleast_2d(np.linspace(data.xlow, data.xup, 1000)).transpose()
        fXpred = np.zeros((1000, data.dim))
        for j in range(1000):
            fXpred[j] = data.objfunction(Xpred[j, :]) + 3*np.cos(10000*Xpred[j, :])

        plt.plot(Xpred, rbf1.evals(Xpred), c='b', linewidth=1.0, zorder=2)
        plt.hold('on')
        plt.plot(Xpred, rbf2.evals(Xpred), c='r', linewidth=1.0, zorder=1)
        plt.scatter(X, fX, c='k', s=10)
        plt.ylim([-35, 45])
        plt.show()
        """

if __name__ == '__main__':
    main()
