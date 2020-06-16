import numpy as np

from ..utils import round_vars
from .candidate_srbf import weighted_distance_merit


def candidate_uniform(num_pts, opt_prob, surrogate, X, fX, weights, Xpend=None, subset=None, dtol=1e-3, num_cand=None):
    """Select new evaluations from uniform candidate points.

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
    :param weights: num_pts weights in [0, 1] for merit function
    :type weights: list or numpy.array
    :param Xpend: Pending evaluations
    :type Xpend: numpy.array
    :param subset: Coordinates that should be perturbed, use None for all
    :type subset: list or numpy.array
    :param dtol: Minimum distance between evaluated and pending points
    :type dtol: float
    :param num_cand: Number of candidate points
    :type num_cand: int

    :return: The num_pts new points to evaluate
    :rtype: numpy.array of size num_pts x dim
    """
    # Find best solution
    xbest = np.copy(X[np.argmin(fX), :]).ravel()

    # Fix default values
    if num_cand is None:
        num_cand = 100 * opt_prob.dim
    if subset is None:
        subset = np.arange(0, opt_prob.dim)

    # Generate uniformly random candidate points
    cand = np.multiply(np.ones((num_cand, opt_prob.dim)), xbest)
    cand[:, subset] = np.random.uniform(opt_prob.lb[subset], opt_prob.ub[subset], (num_cand, len(subset)))

    # Round integer variables
    cand = round_vars(cand, opt_prob.int_var, opt_prob.lb, opt_prob.ub)

    # Make selections
    return weighted_distance_merit(
        num_pts=num_pts, surrogate=surrogate, X=X, fX=fX, Xpend=Xpend, cand=cand, dtol=dtol, weights=weights
    )
