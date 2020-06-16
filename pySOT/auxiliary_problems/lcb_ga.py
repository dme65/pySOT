import numpy as np

from ..utils import GeneticAlgorithm as GA
from ..utils import round_vars
from .lcb_merit import lcb_merit


def lcb_ga(num_pts, opt_prob, surrogate, X, fX, Xpend=None, kappa=2.0, dtol=1e-3, lcb_target=None):
    """Minimize the LCB using a genetic algorithm.

    :param num_pts: Number of points to generate
    :type num_pts: int
    :param opt_prob: Optimization problem
    :type opt_prob: object
    :param surrogate: Surrogate model object
    :type surrogate: object
    :param X: Previously evaluated points, of size n x dim
    :type X: numpy.array
    :param fX: Values at previously evaluated points, of size n x 1
    :type fX: numpy.array
    :param Xpend: Pending evaluations
    :type Xpend: numpy.array
    :param dtol: Minimum distance between evaluated and pending points
    :type dtol: float
    :param lcb_target: Return None if we don't find an LCB value <= lcb_target
    :type lcb_target: float

    :return: num_pts new points to evaluate
    :rtype: numpy.array of size num_pts x dim
    """

    if Xpend is None:  # cdist can't handle None arguments
        Xpend = np.empty([0, opt_prob.dim])
    XX = np.vstack((X, Xpend))

    new_points = np.zeros((num_pts, opt_prob.dim))
    for i in range(num_pts):

        def obj(Y):
            """Round integer variables and compute LCB."""
            Y = round_vars(Y.copy(), opt_prob.int_var, opt_prob.lb, opt_prob.ub)
            return lcb_merit(X=Y, surrogate=surrogate, fX=fX, XX=XX, dtol=dtol, kappa=kappa)

        ga = GA(
            function=obj,
            dim=opt_prob.dim,
            lb=opt_prob.lb,
            ub=opt_prob.ub,
            int_var=opt_prob.int_var,
            pop_size=max([2 * opt_prob.dim, 100]),
            num_gen=100,
        )
        x_best, f_min = ga.optimize()

        if f_min > lcb_target:
            return None  # Give up

        new_points[i, :] = x_best
        XX = np.vstack((XX, x_best))
    return new_points
