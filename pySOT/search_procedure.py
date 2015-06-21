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
    - DyCORS: Uses a DDS strategy with a cap on the lowest probability
    - CandidateUniform: Sample the candidate points uniformly in the domain
    - CandidateSRBF_INT: Uses CandidateSRBF but only
        perturbs the integer variables
    - CandidateSRBF_CONT: Uses CandidateSRBF but only
        perturbs the continuous variables
    - CandidateDyCORS_INT: Uses CandidateSRBF but only
        perturbs the integer variables
    - CandidateDyCORS_CONT: Uses CandidateSRBF but only
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

# ========================= Useful helpers =======================


def unit_rescale(xx):
    """Shift and rescale elements of a vector to the unit interval
    """
    xmax = np.amax(xx)
    xmin = np.amin(xx)
    if xmax == xmin:
        return np.ones(xx.shape)
    else:
        return (xx-xmin)/(xmax-xmin)


def point_set_dist(x, xx):
    """Compute the distance from a point to a set
    """
    return np.sqrt(np.min(np.sum((xx - x)**2, axis=1)))


def closest_point(x, xx):
    """Find point in xx closest to x
    """
    ind = np.argmin(np.sum((xx - x)**2, axis=1))
    return xx[ind, :]


def round_vars(data, x):
    """Round integer variables to closest integer
    """
    if len(data.integer) > 0:
        x[:, data.integer] = np.round(x[:, data.integer])
        # Make sure we don't violate the bound constraints
        for i in data.integer:
            ind = np.where(x[:, i] < data.xlow[i])
            x[ind, i] += 1
            ind = np.where(x[:, i] > data.xup[i])
            x[ind, i] -= 1
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

    def init(self, start_sample, avoid=None):
        """Initialize the multi-search strategy

        :param start_sample: Points in the initial design that
            will be evaluated before the adaptive sampling starts
        :param avoid: Points to avoid
        """
        self.avoid = avoid
        self.proposed_points = np.copy(start_sample)
        for i in range(len(self.search_strategies)):
            self.search_strategies[i].init(self.proposed_points, avoid)

    def remove_point(self, x):
        """Remove x from proposed_points. Useful if x was never evaluated.

        :param x: Point to remove
        """
        idx = np.sum(np.abs(self.proposed_points - x), axis=1).argmin()
        self.proposed_points = np.delete(self.proposed_points, idx, axis=0)
        for i in range(len(self.search_strategies)):
            self.search_strategies[i].proposed_points = self.proposed_points

    def make_points(self, xbest, sigma, evals, maxeval, issync):
        """Create new candidate points. This call is ignored by the optimization
        based search strategies.


        :param xbest: Best solution found
        :param sigma: Current stdDev used to generate candidate points
        :param evals: Routine for predicting function values
        :param maxeval: Evaluation budget
        :param issync: Flag that indicates if the run is synchronous or not.
            If we are running a synchronous strategy we generate
            candidate points for all search procedures. If we are
            running an asynchronous strategy we generate
            candidate points only for the next search procedure
        """
        if issync:
            for i in range(len(self.search_strategies)):
                self.search_strategies[i].make_points(xbest, sigma,
                                                      evals, maxeval)
        else:
            weight = self.weights[self.currentWeight]
            self.search_strategies[weight].make_points(xbest, sigma,
                                                       evals, maxeval)

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
    merit = weight*cand.fhvals.ravel() + \
        (1-weight)*unit_rescale(cand.dmerit.ravel())
    merit[cand.dmerit.ravel() < cand.dtol] = np.inf
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

    def __init__(self, data, numcand=None, ):
        """Initialize the multi-search strategy

        :param data: Optimization object
        :param numcand: Number of candidate points to generate
        """
        self.data = data
        self.avoid = None
        self.xrange = np.asarray(data.xup-data.xlow)
        minxrange = np.amin(self.xrange)
        self.dtol = 1e-3 * minxrange * math.sqrt(data.dim)
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

    def init(self, startsample, avoid=None):
        """Add response surface, experimental design and points to avoid

        :param startsample:  Points generated by the experimental design
        :param avoid: Points to avoid
        """
        self.proposed_points = startsample
        self.avoid = avoid

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

    def make_points(self, xbest, sigma, evals, maxeval=None,
                    issync=False, subset=None):
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
        data = self.data
        dim = data.dim
        ncand = self.numcand
        scalefactors = sigma*self.xrange
        # Make sure that the scale factors are correct for
        # the integer variables (at least 1)
        ind = np.intersect1d(self.data.integer, subset)
        if len(ind) > 0:
            scalefactors[ind] = np.maximum(scalefactors[ind], 1)
        xcand = np.zeros((ncand, dim))
        xcand[:, subset] = np.random.randn(ncand, len(subset))
        xcand = np.multiply(xcand, scalefactors)
        xcand = np.reshape(xbest, (1, dim)) + xcand
        xcand = np.minimum(np.reshape(data.xup, (1, dim)), xcand)
        self.xcand = round_vars(data, np.maximum(
            np.reshape(data.xlow, (1, dim)), xcand))

        devals = scp.distance.cdist(self.xcand, self.proposed_points)
        self.dmerit = np.amin(np.asmatrix(devals), axis=1)
        fhvals = evals(self.xcand)
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

    def make_points(self, xbest, sigma, evals, maxeval=None,
                    issync=False, subset=None):
        """Create new candidate points based on the best solution and the current
        value of sigma.

        :param xbest: Ignored by this method
        :param sigma: Ignored by this method
        :param evals: Routine for predicting function values
        :param maxeval: Ignored by this method
        :param issync: Ignored by this method
        :param subset: Dimensions that will be perturbed
        """
        if subset is None:
            subset = np.arange(0, self.data.dim)
        data = self.data
        dim = data.dim
        ncand = self.numcand
        xcand = np.zeros((ncand, dim))
        xcand[:, subset] = np.matrix(np.random.uniform(
            self.data.xlow, self.data.xup,
            (self.numcand, len(subset))))
        self.xcand = round_vars(data, xcand)
        devals = scp.distance.cdist(self.xcand, self.proposed_points)
        self.dmerit = np.amin(np.asmatrix(devals), axis=1)
        fhvals = evals(self.xcand)
        self.fhvals = unit_rescale(fhvals)
        self.xsample = []


class CandidateDyCORS(CandidateSRBF):
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
        """Initialize the DyCORS strategy

        :param data: Optimization object
        :param numcand:  Number of candidate points to generate
        :param constraint_handler: Object for handling non-bound constraints
        """
        CandidateSRBF.__init__(self, data, numcand=numcand)
        self.minprob = np.min([1, 3/self.data.dim])

    def make_points(self, xbest, sigma, evals, maxeval=None,
                    issync=False, subset=None):
        """Create new candidate points based on the best
        solution and the current value of sigma.

        :param xbest: Best solution found so far
        :param sigma: Current radius, i.e. stdDev used
            to generate candidate points
        :param evals: Routine for predicting function values
        :param maxeval: Evaluation budget, used to calculate
            the DDS probability
        :param issync: Ignored by this method
        :param subset: Dimensions that can be perturbed
            (can be picked by DyCORS)
        """

        # By default we allow any dimension to be perturbed
        if subset is None:
            subset = np.arange(0, self.data.dim)
        dim = self.data.dim
        ncand = self.numcand
        numeval = len(self.proposed_points)
        # ddsProb = np.max([self.minprob,
        # 1-np.log(float(numeval+1))/np.log(float(maxeval))])
        ddsprob = 0.5 - np.arctan(-10 + 20*numeval/float(maxeval))/np.pi
        ddsprob = np.max([self.minprob, ddsprob])
        r = np.random.rand(ncand, len(subset))
        ar = (r < ddsprob)
        if len(subset) > 2:
            # Find where less than 2 indices are perturbed
            ind = np.where(np.sum(ar, axis=1) < 2)[0]
            ar[ind, :] = False
            # We now generate two random dimensions to
            # perturb for each dimensions where we
            # perturbed less than 2. We ignore he possibility
            # of perturbing the same dimension twice since
            # this is not necessarily bad
            ar[ind, np.random.randint(0, len(subset)-1, len(ind))] = True
            ar[ind, np.random.randint(0, len(subset)-1, len(ind))] = True

        # Create candidate points
        scalefactors = sigma*self.xrange
        # Make sure that the scale factors are correct for
        # the integer variables (at least 1)
        ind = np.intersect1d(self.data.integer, subset)
        if len(ind) > 0:
            scalefactors[ind] = np.maximum(scalefactors[ind], 1)

        xcand = np.zeros((ncand, dim))
        xcand[:, subset] = np.random.randn(ncand, len(subset))
        xcand = np.multiply(xcand, scalefactors)
        xcand[:, subset] = np.multiply(xcand[:, subset], ar)
        xcand = np.reshape(xbest, (1, dim)) + xcand

        # Reflections
        for i in range(dim):
            ind = np.where(xcand[:, i] > self.data.xup[i])
            xcand[ind, :] = 2*self.data.xup[i] - xcand[ind, :]
            ind = np.where(xcand[:, i] < self.data.xlow[i])
            xcand[ind, :] = 2*self.data.xlow[i] - xcand[ind, :]

        # Projections
        xcand = np.minimum(np.reshape(self.data.xup, (1, dim)), xcand)
        self.xcand = round_vars(self.data, np.maximum(
            np.reshape(self.data.xlow, (1, dim)), xcand))

        devals = scp.distance.cdist(self.xcand, self.proposed_points)
        self.dmerit = np.amin(np.asmatrix(devals), axis=1)
        fhvals = evals(self.xcand)
        self.fhvals = unit_rescale(fhvals)
        self.xsample = []


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
    def make_points(self, xbest, sigma, evals, maxeval=None,
                    issync=False, subset=None):
        if len(self.data.integer) > 0:
            CandidateSRBF.make_points(self, xbest, sigma, evals,
                                      maxeval=maxeval, issync=issync,
                                      subset=self.data.integer)
        else:
            CandidateSRBF.make_points(self, xbest, sigma, evals,
                                      maxeval=maxeval, issync=issync,
                                      subset=self.data.continuous)


class CandidateDyCORS_INT(CandidateDyCORS):
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
    def make_points(self, xbest, sigma, evals, maxeval=None,
                    issync=False, subset=None):
        if len(self.data.integer) > 0:
            CandidateDyCORS.make_points(self, xbest, sigma, evals,
                                        maxeval=maxeval, issync=issync,
                                        subset=self.data.integer)
        else:
            CandidateDyCORS.make_points(self, xbest, sigma, evals,
                                        maxeval=maxeval, issync=issync,
                                        subset=self.data.continuous)


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
    def make_points(self, xbest, sigma, evals, maxeval=None,
                    issync=False, subset=None):
        if len(self.data.integer) > 0:
            CandidateUniform.make_points(self, xbest, sigma, evals,
                                         maxeval=maxeval, issync=issync,
                                         subset=self.data.integer)
        else:
            CandidateUniform.make_points(self, xbest, sigma, evals,
                                         maxeval=maxeval, issync=issync,
                                         subset=self.data.continuous)


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
    def make_points(self, xbest, sigma, evals, maxeval=None,
                    issync=False, subset=None):
        if len(self.data.continuous) > 0:
            CandidateSRBF.make_points(self, xbest, sigma, evals,
                                      maxeval=maxeval, issync=issync,
                                      subset=self.data.continuous)
        else:
            CandidateSRBF.make_points(self, xbest, sigma, evals,
                                      maxeval=maxeval, issync=issync,
                                      subset=self.data.integer)


class CandidateDyCORS_CONT(CandidateDyCORS):
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
    def make_points(self, xbest, sigma, evals, maxeval=None,
                    issync=False, subset=None):
        if len(self.data.continuous) > 0:
            CandidateDyCORS.make_points(self, xbest, sigma, evals,
                                        maxeval=maxeval, issync=issync,
                                        subset=self.data.continuous)
        else:
            CandidateDyCORS.make_points(self, xbest, sigma, evals,
                                        maxeval=maxeval, issync=issync,
                                        subset=self.data.integer)


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

    def make_points(self, xbest, sigma, evals, maxeval=None,
                    issync=False, subset=None):
        if len(self.data.continuous) > 0:
            CandidateUniform.make_points(self, xbest, sigma, evals,
                                         maxeval=maxeval, issync=issync,
                                         subset=self.data.continuous)
        else:
            CandidateUniform.make_points(self, xbest, sigma, evals,
                                         maxeval=maxeval, issync=issync,
                                         subset=self.data.integer)
