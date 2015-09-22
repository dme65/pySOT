"""
.. module:: search_procedure
  :synopsis: ways of finding the next point to evaluate
.. moduleauthor:: David Eriksson <dme65@cornell.edu>
                 David Bindel <bindel@cornell.edu>

:Module: search_procedure
:Author: David Eriksson <dme65@cornell.edu>,
    David Bindel <bindel@cornell.edu>

We currently only support the weighted distance merit function.

We support the following methods for minimizing on the surface:

Candidate based methods:
    - CandidateSRBF: Generate candidate points around the best point
    - CandidateDYCORS: Uses a DDS strategy with a cap on the lowest probability
    - CandidateUniform: Sample the candidate points uniformly in the domain
    - CandidateSRBF_INT: Uses CandidateSRBF but only
        perturbs the integer variables
    - CandidateSRBF_CONT: Uses CandidateSRBF but only
        perturbs the continuous variables
    - CandidateDYCORS_INT: Uses CandidateSRBF but only
        perturbs the integer variables
    - CandidateDYCORS_CONT: Uses CandidateSRBF but only
        perturbs the continuous variables
    - CandidateUniform_CONT: Sample the continuous variables of
        the candidate points uniformly in the domain
    - CandidateUniform_INT: Sample the integer variables of the
        candidate points uniformly in the domain

We also support using multiple of these strategies and cycle
    between them which we call MultiSearchStrategy
    - MultiSearchStrategy
"""

import math
import numpy as np
import scipy.spatial as scp
from heuristic_algorithms import GeneticAlgorithm as GA
from scipy.optimize import minimize
import scipy.stats as stats

# ========================= Useful helpers =======================


def to_unit_box(x, data):
    return (np.copy(x) - data.xlow) / (data.xup - data.xlow)


def from_unit_box(x, data):
        return data.xlow + (data.xup - data.xlow) * np.copy(x)


def unit_rescale(xx):
    """Shift and rescale elements of a vector to the unit interval
    """
    xmax = np.amax(xx)
    xmin = np.amin(xx)
    if xmax == xmin:
        return np.ones(xx.shape)
    else:
        return (xx-xmin)/(xmax-xmin)


def round_vars(data, x):
    """Round integer variables to closest integer
    """
    if len(data.integer) > 0:
        # Map to original range
        x = from_unit_box(x, data)
        # Round the original ranged integer variables
        x[:, data.integer] = np.round(x[:, data.integer])
        # Make sure we don't violate the bound constraints
        for i in data.integer:
            ind = np.where(x[:, i] < data.xlow[i])
            x[ind, i] += 1
            ind = np.where(x[:, i] > data.xup[i])
            x[ind, i] -= 1
        # Map back to [0, 1] and return
        return to_unit_box(x, data)
    return x

# ========================= MultiSearch =======================


class MultiSearchStrategy(object):
    """ A collection of Search Strategies and weights so that the user
        can use multiple search strategies for the same optimization
        problem. This object keeps an internal list of proposed points
        in order to be able to compute the minimum distance from a point
        to all proposed evaluations. This list has to be reset each time
        the optimization algorithm restarts.

        :ivar search_strategies: List of search strategies
        :ivar weights: List of integer weights that will be iterated in
            order to determine what search strategy to use next.
            If for example the weights [0 0 1] are used, then
            search strategy 0 will be used two times, then
            strategy 1 is used after which the cycling restarts.
        :ivar proposed_points: List of all points proposed by this strategy
        :ivar currentWeight: Current weights
        :ivar data: Optimization problem object
        :ivar avoid: Points to avoid
    """
    def __init__(self, strategy_list, weight_strategy):
        # Check so that the weights are correct
        if weight_strategy is None:
            weight_strategy = range(len(strategy_list))
        if (not all(isinstance(ii, int) for ii in weight_strategy)) or \
                np.min(weight_strategy) < 0 or \
                np.max(weight_strategy) > len(strategy_list)-1:
            raise ValueError("Incorrect weights!!")
        self.search_strategies = strategy_list
        self.weights = weight_strategy
        self.currentWeight = 0
        self.proposed_points = None
        self.data = strategy_list[0].data
        self.avoid = None
        self.budget = None
        self.n0 = None
        self.issync = None
        self.fhat = None

    def init(self, start_sample, budget, issync, fhat, avoid=None):
        """Initialize the multi-search strategy

        :param start_sample: Points in the initial design that
            will be evaluated before the adaptive sampling starts
        :param avoid: Points to avoid
        """
        self.proposed_points = start_sample
        self.avoid = avoid
        self.n0 = start_sample.shape[0]
        self.budget = budget
        self.fhat = fhat
        self.issync = issync
        for i in range(len(self.search_strategies)):
            self.search_strategies[i].init(self.proposed_points, budget, issync, fhat, avoid)

    def remove_point(self, x):
        """Remove x from proposed_points. Useful if x was never evaluated.

        :param x: Point to remove
        """
        idx = np.sum(np.abs(self.proposed_points - x), axis=1).argmin()
        self.proposed_points = np.delete(self.proposed_points, idx, axis=0)
        for i in range(len(self.search_strategies)):
            self.search_strategies[i].proposed_points = self.proposed_points

    def make_points(self, xbest, sigma, evals, derivs):
        """Create new candidate points. This call is ignored by the optimization
        based search strategies.


        :param xbest: Best solution found
        :param sigma: Current stdDev used to generate candidate points
        :param evals: Routine for predicting function values
        :param deriv: Routine for predicting derivatives
        """
        if self.issync:
            for i in range(len(self.search_strategies)):
                self.search_strategies[i].make_points(xbest, sigma, evals, derivs)
        else:
            weight = self.weights[self.currentWeight]
            self.search_strategies[weight].make_points(xbest, sigma, evals, derivs)

    def next(self):
        """Generate the next proposed point from the current search strategy,
        update the list of proposed points and move the counter forward to
        the next search strategy

        :return: Next point to evaluate
        """
        weight = self.weights[self.currentWeight]
        xnew = np.reshape(self.search_strategies[weight].next(),
                          (-1, self.data.dim))
        self.proposed_points = self.search_strategies[weight].proposed_points
        # Update samples
        for i in range(len(self.search_strategies)):
            self.search_strategies[i].proposed_points = self.proposed_points
            # Append xsample to all other candidate based methods
            if self.search_strategies[i].usecand and i != weight:
                if self.search_strategies[i].xsample is None:
                    self.search_strategies[i].xsample = []
                    self.search_strategies[i].xsample.append(xnew)
                else:
                    self.search_strategies[i].xsample.append(xnew)

        self.currentWeight = (self.currentWeight + 1) % len(self.weights)

        return xnew


# ====================== Candidate based search methods =====================

def candidate_merit_weighted_distance(cand):
    """Weighted distance merit functions for the candidate points based methods
    """

    ii = cand.nextWeight
    if cand.xsample:
        ds = scp.distance.cdist(cand.xcand, np.atleast_2d(cand.xsample[-1]))
        cand.dmerit = np.minimum(cand.dmerit, ds)
    if cand.avoid:
        for _, xavoid in cand.avoid.iteritems():
            xavoid = np.reshape(xavoid, (1, xavoid.shape[0]))
            davoid = scp.distance.cdist(cand.xcand, xavoid)
            cand.dmerit = np.minimum(cand.dmerit, davoid)

    weight = cand.weights[(ii+len(cand.weights)) % len(cand.weights)]
    merit = weight*cand.fhvals + \
        (1-weight)*(1.0 - unit_rescale(cand.dmerit))
    merit[cand.dmerit < cand.dtol] = np.inf
    jj = np.argmin(merit)
    cand.fhvals[jj] = np.inf
    cand.xsample.append(cand.xcand[jj, :])
    cand.nextWeight += 1
    return cand.xcand[jj, :]


class CandidateSRBF(object):
    """This is an implementation of the candidate points method that is
    proposed in the first SRBF paper. Candidate points are generated
    by making normally distributed perturbations with stdDev sigma
    around the best solution

    :ivar usecand: Indicates that this method is candidate based
    :ivar data: Optimization object
    :ivar weights: Weights used in the merit function
    :ivar numcand: Number of candidate points to generate
    :ivar xsample: The proposed evaluations since
    :ivar proposed_points: List of points proposed by any search strategy
        since the last restart
    """

    usecand = True

    def __init__(self, data, numcand=None):
        """Initialize the multi-search strategy

        :param data: Optimization object
        :param numcand: Number of candidate points to generate
        """
        self.data = data
        self.fhat = None
        self.avoid = None
        self.xrange = np.ones(data.xlow.shape)
        self.dtol = 1e-3 * math.sqrt(data.dim)
        self.weights = np.array([0.3, 0.5, 0.8, 0.95])
        self.dmerit = None
        self.fhvals = None
        self.xsample = None
        self.xcand = None
        self.proposed_points = None
        self.nextWeight = 0
        self.numcand = numcand
        if self.numcand is None:
            self.numcand = min([5000, 100*data.dim])
        self.budget = None
        self.n0 = None
        self.issync = None

    def init(self, start_sample, budget, issync, fhat, avoid=None):
        self.proposed_points = start_sample
        self.avoid = avoid
        self.n0 = start_sample.shape[0]
        self.budget = budget
        self.fhat = fhat
        self.issync = issync

    def remove_point(self, x):
        """Remove x from the list of proposed points.
        Useful if x was never evaluated.

        :param x: Point to be removed
        """
        idx = np.sum(np.abs(self.proposed_points - x), axis=1).argmin()
        self.proposed_points = np.delete(self.proposed_points, idx, axis=0)

    def next(self):
        """Propose a new point to evaluate and update list of proposed points

        :return: Next point to evaluate
        """
        xnew = candidate_merit_weighted_distance(self)  # FIXME, hard-coded
        self.proposed_points = np.vstack((self.proposed_points,
                                          np.asarray(xnew)))
        return xnew

    def _generate_cand(self, scalefactors, xbest, subset):
        xcand = np.ones((self.numcand,  self.data.dim)) * xbest
        for i in subset:
            lower, upper = 0.0, 1.0
            ssigma = scalefactors[i]
            xcand[:, i] = stats.truncnorm.rvs(
                (lower - xbest[i]) / ssigma, (upper - xbest[i]) / ssigma,
                loc=xbest[i], scale=ssigma, size=self.numcand)
        self.xcand = round_vars(self.data, xcand)

    def make_points(self, xbest, sigma, evals, derivs=None, subset=None):
        """Create new candidate points based on the best
        solution and the current value of sigma.

        :param xbest: Best solution found so far
        :param sigma: Current radius, i.e. stdDev used
            to generate candidate points
        :param maxeval: Ignored by this method
        :param issync: Ignored by this method
        :param subset: Dimensions that will be perturbed
        """

        if subset is None:
            subset = np.arange(0, self.data.dim)
        scalefactors = sigma * self.xrange

        # Make sure that the scale factors are correct for
        # the integer variables (at least 1)
        ind = np.intersect1d(self.data.integer, subset)
        if len(ind) > 0:
            scalefactors[ind] = np.maximum(scalefactors[ind], 1.0 / (self.data.xup[ind] - self.data.xlow[ind]))

        # Generate candidate points
        self._generate_cand(scalefactors, xbest, subset)

        devals = scp.distance.cdist(self.xcand, self.proposed_points)

        # Since the response surface may have the centers in a  different order than the
        # evaluations were proposed we need to map the distances to the centers correctly
        mapping = []
        for i in range(self.fhat.nump):
            mapping.append(np.sum(np.abs(self.proposed_points - self.fhat.x[i, :]), axis=1).argmin())

        fhvals = evals(self.xcand, devals[:, mapping].T, scaling=True)

        self.dmerit = np.amin(np.asmatrix(devals), axis=1)
        self.fhvals = unit_rescale(fhvals)
        self.xsample = []


class CandidateUniform(CandidateSRBF):
    """Create Candidate points by sampling uniformly in the domain

    :ivar usecand: Indicates that this method is candidate based
    :ivar data: Optimization object
    :ivar weights: Weights used in the merit function
    :ivar numcand: Number of candidate points to generate
    :ivar xsample: The proposed evaluations since
    :ivar proposed_points: List of points proposed by any search strategy
        since the last restart
    """

    def _generate_cand(self, scalefactors, xbest, subset):
        xcand = np.ones((self.numcand, self.data.dim)) * xbest
        xcand[:, subset] = np.random.uniform(
            0.0, 1.0, (self.numcand, len(subset)))

        self.xcand = round_vars(self.data, xcand)


class CandidateDYCORS(CandidateSRBF):
    """This is an implementation of DyCORS method to generate
    candidate points. The DyCORS method uses the DDS algorithm
    which only perturbs a subset of the dimensions when
    perturbing the best solution. The probability for a
    dimension to be perturbed decreases after each
    evaluation and is capped in order to guarantee
    global convergence.

    :ivar usecand: Indicates that this method is candidate based
    :ivar data: Optimization object
    :ivar weights: Weights used in the merit function
    :ivar numcand: Number of candidate points to generate
    :ivar xsample: The proposed evaluations since
    :ivar proposed_points: List of points proposed by any search strategy
        since the last restart
    """
    def __init__(self, data, numcand=None):
        """Initialize the DYCORS strategy

        :param data: Optimization object
        :param numcand:  Number of candidate points to generate
        """
        CandidateSRBF.__init__(self, data, numcand=numcand)
        self.minprob = np.min([1.0, 1.0/self.data.dim])

        def probfun(numevals, budget):
            return min([20.0/data.dim, 1.0]) * (1.0-(np.log(numevals + 1.0) / np.log(budget)))
        self.probfun = probfun

    def _generate_cand(self, scalefactors, xbest, subset):
        ddsprob = self.probfun(self.proposed_points.shape[0] - self.n0, self.budget - self.n0)
        ddsprob = np.max([self.minprob, ddsprob])

        nlen = len(subset)
        ar = (np.random.rand(self.numcand, nlen) < ddsprob)
        ind = np.where(np.sum(ar, axis=1) == 0)[0]
        ar[ind, np.random.randint(0, nlen - 1, size=len(ind))] = 1

        xcand = np.ones((self.numcand, self.data.dim)) * xbest
        for i in range(nlen):
            lower, upper = 0.0, 1.0
            ssigma = scalefactors[subset[i]]
            ind = np.where(ar[:, i] == 1)[0]
            xcand[ind, subset[i]] = stats.truncnorm.rvs(
                (lower - xbest[subset[i]]) / ssigma, (upper - xbest[subset[i]]) / ssigma,
                loc=xbest[subset[i]], scale=ssigma, size=len(ind))

        self.xcand = round_vars(self.data, xcand)


class CandidateDDS(CandidateDYCORS):
    def __init__(self, data, numcand=None):
        CandidateDYCORS.__init__(self, data, numcand=numcand)
        self.weights = np.array([1.0])
        self.numcand = max([0.5*data.dim, 2])

        def probfun(numevals, budget):
            return 1.0-(np.log(numevals + 1.0) / np.log(budget))
        self.probfun = probfun

    def make_points(self, xbest, sigma, evals, derivs=None, subset=None):
        CandidateDYCORS.make_points(self, xbest, 0.2, evals, derivs, subset)


class CandidateSRBF_INT(CandidateSRBF):
    """Candidate points are generated by perturbing
    ONLY the discrete variables using the SRBF strategy

    :ivar usecand: Indicates that this method is candidate based
    :ivar data: Optimization object
    :ivar weights: Weights used in the merit function
    :ivar numcand: Number of candidate points to generate
    :ivar xsample: The proposed evaluations since
    :ivar proposed_points: List of points proposed by any search strategy
        since the last restart
    """
    def make_points(self, xbest, sigma, evals, derivs=None, subset=None):
        if len(self.data.integer) > 0:
            CandidateSRBF.make_points(self, xbest, sigma,
                                      evals, derivs, subset=self.data.integer)
        else:
            CandidateSRBF.make_points(self, xbest, sigma,
                                      evals, derivs, subset=self.data.continuous)


class CandidateDYCORS_INT(CandidateDYCORS):
    """Candidate points are generated by perturbing ONLY the discrete
    variables using the DyCORS strategy

    :ivar usecand: Indicates that this method is candidate based
    :ivar data: Optimization object
    :ivar weights: Weights used in the merit function
    :ivar numcand: Number of candidate points to generate
    :ivar xsample: The proposed evaluations since
    :ivar proposed_points: List of points proposed by any search strategy
        since the last restart
    """
    def make_points(self, xbest, sigma, evals, derivs=None, subset=None):
        if len(self.data.integer) > 0:
            CandidateDYCORS.make_points(self, xbest, sigma,
                                        evals, derivs=None, subset=self.data.integer)
        else:
            CandidateDYCORS.make_points(self, xbest, sigma,
                                        evals, derivs=None, subset=self.data.continuous)


class CandidateUniform_INT(CandidateUniform):
    """Candidate points are generated by perturbing ONLY
    the discrete variables using the uniform perturbations

    :ivar usecand: Indicates that this method is candidate based
    :ivar data: Optimization object
    :ivar weights: Weights used in the merit function
    :ivar numcand: Number of candidate points to generate
    :ivar xsample: The proposed evaluations since
    :ivar proposed_points: List of points proposed by any search strategy
        since the last restart
    """
    def make_points(self, xbest, sigma, evals, derivs=None, subset=None):
        if len(self.data.integer) > 0:
            CandidateUniform.make_points(self, xbest, sigma, evals,
                                         derivs=None, subset=self.data.integer)
        else:
            CandidateUniform.make_points(self, xbest, sigma, evals,
                                         derivs=None, subset=self.data.continuous)


class CandidateSRBF_CONT(CandidateSRBF):
    """Candidate points are generated by perturbing ONLY
    the continuous variables using the SRBF strategy

    :ivar usecand: Indicates that this method is candidate based
    :ivar data: Optimization object
    :ivar weights: Weights used in the merit function
    :ivar numcand: Number of candidate points to generate
    :ivar xsample: The proposed evaluations since
    :ivar proposed_points: List of points proposed by any search strategy
        since the last restart
    """
    def make_points(self, xbest, sigma, evals, derivs=None, subset=None):
        if len(self.data.continuous) > 0:
            CandidateSRBF.make_points(self, xbest, sigma, evals,
                                      derivs=None, subset=self.data.continuous)
        else:
            CandidateSRBF.make_points(self, xbest, sigma, evals,
                                      derivs=None, subset=self.data.integer)


class CandidateDYCORS_CONT(CandidateDYCORS):
    """Candidate points are generated by perturbing ONLY
    the continuous variables using the DyCORS strategy

    :ivar usecand: Indicates that this method is candidate based
    :ivar data: Optimization object
    :ivar weights: Weights used in the merit function
    :ivar numcand: Number of candidate points to generate
    :ivar xsample: The proposed evaluations since
    :ivar proposed_points: List of points proposed by any search strategy
        since the last restart
    """
    def make_points(self, xbest, sigma, evals, derivs=None, subset=None):
        if len(self.data.continuous) > 0:
            CandidateDYCORS.make_points(self, xbest, sigma, evals,
                                        derivs=None, subset=self.data.continuous)
        else:
            CandidateDYCORS.make_points(self, xbest, sigma, evals,
                                        derivs=None, subset=self.data.integer)


class CandidateUniform_CONT(CandidateUniform):
    """Candidate points are generated by perturbing ONLY
    the continuous variables using uniform perturbations

    :ivar usecand: Indicates that this method is candidate based
    :ivar data: Optimization object
    :ivar weights: Weights used in the merit function
    :ivar numcand: Number of candidate points to generate
    :ivar xsample: The proposed evaluations since
    :ivar proposed_points: List of points proposed by any search strategy
        since the last restart
    """

    def make_points(self, xbest, sigma, evals, derivs=None, subset=None):
        if len(self.data.continuous) > 0:
            CandidateUniform.make_points(self, xbest, sigma, evals,
                                         derivs=None, subset=self.data.continuous)
        else:
            CandidateUniform.make_points(self, xbest, sigma, evals,
                                         derivs=None, subset=self.data.integer)

################################## Optimization based strategies ###################################

class GeneticAlgorithm(object):

    usecand = False

    def __init__(self, data):
        """Initialize the multi-search strategy

        :param data: Optimization object
        :param numcand: Number of candidate points to generate
        """
        self.data = data
        self.fhat = None
        self.avoid = None
        self.dtol = 1e-3 * math.sqrt(data.dim)
        self.dmerit = None
        self.proposed_points = None
        self.evals = None
        self.derivs = None
        self.budget = None
        self.numevals = None
        self.issync = None

    def init(self, start_sample, budget, issync, fhat, avoid=None):
        self.proposed_points = start_sample
        self.avoid = avoid
        self.budget = budget
        self.numevals = 0
        self.fhat = fhat
        self.issync = issync

    def remove_point(self, x):
        idx = np.sum(np.abs(self.proposed_points - x), axis=1).argmin()
        self.proposed_points = np.delete(self.proposed_points, idx, axis=0)

    def next(self, objfunction=None):

        # Find a new point
        if objfunction is None:
            def objfunction(x):
                return self.evals(to_unit_box(x, self.data), scaling=False)

        ga = GA(objfunction, self.data.dim, self.data.xlow,
                self.data.xup, popsize=max([2*self.data.dim, 100]), ngen=100)
        xnew, fbest = ga.optimize()
        xnew = to_unit_box(xnew, self.data)

        dist = np.atleast_2d(np.min(
            scp.distance.cdist(self.proposed_points, np.atleast_2d(xnew)), axis=1)).T

        if np.min(dist) < self.dtol:
            # Perturb the best solution until we satisfy the tolerance
            d = 0.0
            xbest = None
            while d < self.dtol:
                xbest = xnew + self.dtol * np.random.randn(1, self.data.dim)
                xbest = np.maximum(xbest, np.zeros(self.data.dim))
                xbest = np.minimum(xbest, np.ones(self.data.dim))
                d = np.atleast_2d(np.min(
                    scp.distance.cdist(self.proposed_points, np.atleast_2d(xbest)), axis=1)).T.min()
            xnew = xbest

        self.proposed_points = np.vstack((self.proposed_points, np.asarray(xnew)))
        return xnew

    def make_points(self, xbest, sigma, evals, derivs=None):
        self.evals = evals
        self.derivs = derivs


class MultiStartGradient(object):

    usecand = False

    """ A wrapper around the scipy.optimize implementations of box-constrained
        gradient based minimization.

    Attributes:
        usecand: Indicates that this method is NOT candidate based
        data: Optimization object
        fhat: Original response surface object.
        proposed_points: List of points proposed by any search strategy
                         since the last restart
        objfun: The merit function to minimize (needs to have a gradient available)
        bounds: n x 2 matrix with lower and upper bound constraints
        numrestarts: Number of random starting points
        method: What optimization method to use. The following options are available:
            - L-BFGS-B: Quasi-Newton method of
                        Broyden, Fletcher, Goldfarb, and Shanno (BFGS)
            - TNC:      Truncated Newton algorithm

        Note: SLSQP is supposed to work with bound constraints but for some reason it
              sometimes violates the constraints anyway.
    """
    def __init__(self, data, method='L-BFGS-B', numrestarts=30):
        """Initialize the Multi-Start Gradient object

        Args:
            data: Optimization object
            method: What optimization method to use (see above)
            numrestarts: Number of random starting points
        """
        self.data = data
        self.fhat = None
        self.avoid = None
        self.dtol = 1e-3 * math.sqrt(data.dim)
        self.proposed_points = None
        self.evals = None
        self.derivs = None
        self.budget = None
        self.numevals = None
        self.issync = None
        self.bounds = np.zeros((self.data.dim, 2))
        self.bounds[:, 0] = self.data.xlow
        self.bounds[:, 1] = self.data.xup
        self.numrestarts = numrestarts
        self.xbest = None
        if (method == 'TNC') or (method == 'L-BFGS-B'):
            self.method = method
        else:
            self.method = 'L-BFGS-B'

    def init(self, start_sample, budget, issync, fhat, avoid=None):
        self.proposed_points = start_sample
        self.avoid = avoid
        self.budget = budget
        self.numevals = 0
        self.fhat = fhat
        self.issync = issync

    def remove_point(self, x):
        """ Remove x from self.porposed_points."""
        idx = np.sum(np.abs(self.proposed_points - x), axis=1).argmin()
        self.proposed_points = np.delete(self.proposed_points, idx, axis=0)

    def make_points(self, xbest, sigma, evals, derivs):
        """The method doesn't use candidate points so this method is not used """
        self.evals = evals
        self.derivs = derivs
        self.xbest = xbest

    def next(self):
        """Propose a new point to evaluate and update list of proposed points """
        def eval(x):
            return self.evals(np.atleast_2d(to_unit_box(x, self.data)))[0, 0]

        def deriv(x):
            return self.derivs(to_unit_box(x, self.data)).ravel()

        self.fvals = np.zeros(self.numrestarts)
        self.xvals = np.zeros((self.numrestarts, self.data.dim))
        self.dists = np.zeros(self.numrestarts)

        for i in range(self.numrestarts):
            if i == 0:
                x0 = np.array(from_unit_box(self.xbest, self.data))
            else:
                x0 = np.random.uniform(self.data.xlow, self.data.xup)

            res = minimize(eval, x0, method=self.method,
                           jac=deriv, bounds=self.bounds)

            # Compute the distance to the proposed points
            xx = np.atleast_2d(res.x)
            xx = to_unit_box(xx, self.data)
            dist = np.atleast_2d(np.min(
                scp.distance.cdist(self.proposed_points, xx), axis=1)).T

            self.fvals[i] = res.fun
            self.xvals[i, :] = xx
            self.dists[i] = dist.min()
        if self.dists.max() > self.dtol:
            xbest = None
            fbest = np.inf
            for i in range(self.numrestarts):
                if self.dists[i] > self.dtol and self.fvals[i] < fbest:
                    xbest = self.xvals[i, :]
                    fbest = self.fvals[i]
        else:
            # Perturb the best point
            d = 0.0
            while d < self.dtol:
                xbest = self.xvals[np.argmin(self.fvals), :] + self.dtol * np.random.randn(1, self.data.dim)
                xbest = np.maximum(xbest, np.zeros(self.data.dim))
                xbest = np.minimum(xbest, np.ones(self.data.dim))
                d = np.atleast_2d(np.min(
                    scp.distance.cdist(self.proposed_points, np.atleast_2d(xbest)), axis=1)).T.min()

        self.proposed_points = np.vstack((self.proposed_points, np.asarray(xbest)))
        return xbest.ravel()

if __name__ == "__main__":
    # Test DYCORS
    dim = 10
    maxeval = 100
    initeval = 80
    from test_problems import Ackley
    from experimental_design import LatinHypercube
    from rbf_interpolant import RBFInterpolant
    from rbf_surfaces import CubicRBFSurface
    xbest = np.ones(dim)
    data = Ackley(dim)
    cand = CandidateDYCORS(data, 100*dim)
    exp_des = LatinHypercube(dim, npts=initeval)
    initpoints = exp_des.generate_points()
    rbf = RBFInterpolant(surftype=CubicRBFSurface)
    for i in range(initeval):
        rbf.add_point(initpoints[i, :], data.objfunction(initpoints[i, :]))
    cand.init(initpoints, maxeval, True, None)

    def evals(x, scaling):
        return rbf.evals(x)

    cand.make_points(xbest, 0.02, evals, maxeval)

    import matplotlib.pyplot as plt
    plt.plot(cand.xcand[:, 0], cand.xcand[:, 1], 'ro')
    plt.ylabel('Dimension 1')
    plt.xlabel('Dimension 2')
    plt.show()
