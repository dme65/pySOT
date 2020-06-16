import numpy as np
import scipy.spatial as scpspatial
import scipy.stats as stats

from ..utils import round_vars, unit_rescale


def weighted_distance_merit(num_pts, surrogate, X, fX, cand, weights, Xpend=None, dtol=1e-3):
    """Compute the weighted distance merit function.

    :param num_pts: Number of points to generate
    :type num_pts: int
    :param surrogate: Surrogate model object
    :type surrogate: object
    :param X: Previously evaluated points, of size n x dim
    :type X: numpy.array
    :param fX: Values at previously evaluated points, of size n x 1
    :type fX: numpy.array
    :param cand: Candidate points to select from, of size m x dim
    :type cand: numpy.array
    :param weights: num_pts weights in [0, 1] for merit function
    :type weights: list or numpy.array
    :param Xpend: Pending evaluation, of size k x dim
    :type Xpend: numpy.array
    :param dtol: Minimum distance between evaluated and pending points
    :type dtol: float

    :return: The num_pts new points chosen from the candidate points
    :rtype: numpy.array of size num_pts x dim
    """
    # Distance
    dim = X.shape[1]
    if Xpend is None:  # cdist can't handle None arguments
        Xpend = np.empty([0, dim])
    dists = scpspatial.distance.cdist(cand, np.vstack((X, Xpend)))
    dmerit = np.amin(dists, axis=1, keepdims=True)

    # Values
    fvals = surrogate.predict(cand)
    fvals = unit_rescale(fvals)

    # Pick candidate points
    new_points = np.ones((num_pts, dim))
    for i in range(num_pts):
        w = weights[i]
        merit = w * fvals + (1.0 - w) * (1.0 - unit_rescale(np.copy(dmerit)))

        merit[dmerit < dtol] = np.inf
        jj = np.argmin(merit)
        fvals[jj] = np.inf
        new_points[i, :] = cand[jj, :].copy()

        # Update distances and weights
        ds = scpspatial.distance.cdist(cand, np.atleast_2d(new_points[i, :]))
        dmerit = np.minimum(dmerit, ds)

    return new_points


def candidate_srbf(
    num_pts, opt_prob, surrogate, X, fX, weights, Xpend=None, sampling_radius=0.2, subset=None, dtol=1e-3, num_cand=None
):
    """Select new evaluations using Stochastic RBF (SRBF).

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
    :param Xpend: Pending evaluation, of size k x dim
    :type Xpend: numpy.array
    :param sampling_radius: Perturbation radius
    :type sampling_radius: float
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

    # Compute scale factors for each dimension and make sure they
    # are correct for integer variables (at least 1)
    scalefactors = sampling_radius * (opt_prob.ub - opt_prob.lb)
    ind = np.intersect1d(opt_prob.int_var, subset)
    if len(ind) > 0:
        scalefactors[ind] = np.maximum(scalefactors[ind], 1.0)

    # Generate candidate points
    cand = np.multiply(np.ones((num_cand, opt_prob.dim)), xbest)
    for i in subset:
        lower, upper, sigma = opt_prob.lb[i], opt_prob.ub[i], scalefactors[i]
        cand[:, i] = stats.truncnorm.rvs(
            a=(lower - xbest[i]) / sigma, b=(upper - xbest[i]) / sigma, loc=xbest[i], scale=sigma, size=num_cand
        )

    # Round integer variables
    cand = round_vars(cand, opt_prob.int_var, opt_prob.lb, opt_prob.ub)

    # Make selections
    return weighted_distance_merit(
        num_pts=num_pts, surrogate=surrogate, X=X, fX=fX, Xpend=Xpend, cand=cand, dtol=dtol, weights=weights
    )
