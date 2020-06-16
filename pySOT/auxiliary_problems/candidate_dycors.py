import numpy as np
import scipy.stats as stats

from ..utils import round_vars
from .candidate_srbf import weighted_distance_merit


def candidate_dycors(
    num_pts,
    opt_prob,
    surrogate,
    X,
    fX,
    weights,
    prob_perturb,
    Xpend=None,
    sampling_radius=0.2,
    subset=None,
    dtol=1e-3,
    num_cand=None,
    xbest=None,
):
    """Select new evaluations using DYCORS.

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
    :param prob_perturb: Probability to perturb a given coordinate
    :type prob_perturb: list or numpy.array
    :param Xpend: Pending evaluations
    :type Xpend: numpy.array
    :param sampling_radius: Perturbation radius
    :type sampling_radius: float
    :param subset: Coordinates that should be perturbed, use None for all
    :type subset: list or numpy.array
    :param dtol: Minimum distance between evaluated and pending points
    :type dtol: float
    :param num_cand: Number of candidate points
    :type num_cand: int
    :param xbest: The point around which candidates are generated
    :type xbest: numpy.array

    :return: The num_pts new points to evaluate
    :rtype: numpy.array of size num_pts x dim
    """
    # Find best solution
    if xbest is None:
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
    if len(subset) == 1:  # Fix when nlen is 1
        ar = np.ones((num_cand, 1))
    else:
        ar = np.random.rand(num_cand, len(subset)) < prob_perturb
        ind = np.where(np.sum(ar, axis=1) == 0)[0]
        ar[ind, np.random.randint(0, len(subset) - 1, size=len(ind))] = 1

    cand = np.multiply(np.ones((num_cand, opt_prob.dim)), xbest)
    for i in subset:
        lower, upper, sigma = opt_prob.lb[i], opt_prob.ub[i], scalefactors[i]
        ind = np.where(ar[:, i] == 1)[0]
        cand[ind, subset[i]] = stats.truncnorm.rvs(
            a=(lower - xbest[i]) / sigma, b=(upper - xbest[i]) / sigma, loc=xbest[i], scale=sigma, size=len(ind)
        )

    # Round integer variables
    cand = round_vars(cand, opt_prob.int_var, opt_prob.lb, opt_prob.ub)

    # Make selections
    return weighted_distance_merit(
        num_pts=num_pts, surrogate=surrogate, X=X, fX=fX, Xpend=Xpend, cand=cand, dtol=dtol, weights=weights
    )
