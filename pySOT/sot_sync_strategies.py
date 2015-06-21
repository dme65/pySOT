"""
.. module:: sot_sync_strategies
   :synopsis: Parallel synchronous optimization strategy
.. moduleauthor:: David Bindel <bindel@cornell.edu>,
    David Eriksson <dme65@cornell.edu>

:Module: sot_sync_strategies
:Author: David Bindel <bindel@cornell.edu>,
    David Eriksson <dme65@cornell.edu>

Synchronous strategies for Stochastic RBF
"""

from __future__ import print_function
import sys
import numpy as np
import math
from experimental_design import LatinHypercube
from search_procedure import round_vars, CandidateDyCORS
from poap.strategy import BaseStrategy, RetryStrategy
from rbf_interpolant import phi_cubic, dphi_cubic, linear_tail, \
    dlinear_tail, RBFInterpolant


class SyncStrategyNoConstraints(BaseStrategy):
    """Parallel synchronous optimization strategy without non-bound constraints.

    This class implements the parallel synchronous SRBF strategy
    described by Regis and Shoemaker.  After the initial experimental
    design (which is embarrassingly parallel), the optimization
    proceeds in phases.  During each phase, we allow nsamples
    simultaneous function evaluations.  We insist that these
    evaluations run to completion -- if one fails for whatever reason,
    we will resubmit it.  Samples are drawn randomly from around the
    current best point, and are sorted according to a merit function
    based on distance to other sample points and predicted function
    values according to the response surface.  After several
    successive significant improvements, we increase the sampling
    radius; after several failures to improve the function value, we
    decrease the sampling radius.  We restart once the sampling radius
    decreases below a threshold.
    """

    def __init__(self, worker_id, data, response_surface, maxeval, nsamples,
                 exp_design=None, search_procedure=None, extra=None,
                 quiet=False, stream=sys.stdout):
        """Initialize the optimization strategy.

        :param worker_id: Start ID in a multistart setting
        :param data: Problem parameter data structure
        :param response_surface: Surrogate model object
        :param maxeval: Function evaluation budget
        :param nsamples: Number of simultaneous fevals allowed
        :param exp_design: Experimental design
        :param search_procedure: Search procedure for finding
            points to evaluate
        :param extra: Points to be added to the experimental design
        :param quiet: If True, nothing is printed to the stream
        :param stream: Where progress should be printed, sys.stdout is default
        """

        self.worker_id = worker_id
        self.quiet = quiet
        self.stream = stream
        self.data = data
        self.fhat = response_surface
        if self.fhat is None:
            self.fhat = RBFInterpolant(phi=phi_cubic, P=linear_tail,
                                       dphi=dphi_cubic, dP=dlinear_tail,
                                       eta=1e-8, maxp=maxeval)
        self.maxeval = maxeval
        self.nsamples = nsamples
        self.extra = extra

        # Default to generate sampling points using Symmetric Latin Hypercube
        self.design = exp_design
        if self.design is None:
            self.design = LatinHypercube(data.dim, 2*data.dim+1)

        self.xrange = np.asarray(data.xup - data.xlow)

        # algorithm parameters
        self.sigma_max = 0.2  	# w.r.t. unit box
        self.sigma_min = 0.005  # w.r.t. unit box
        self.failtol = max(5, data.dim)
        self.succtol = 3

        self.numeval = 0
        self.status = 0
        self.sigma = 0
        self.resubmitter = RetryStrategy()
        self.xbest = None
        self.fbest = np.inf
        self.fbest_old = None

        # Set up search procedures and initialize
        self.search = search_procedure
        if self.search is None:
            self.search = CandidateDyCORS(data, numcand=100*data.dim)

        # Start with first experimental design
        self.sample_initial()

    def log(self, message):
        """Record a message string to the log.

        :param message: Message to be printed to the logfile
        """
        if not self.quiet:
            print(message, file=self.stream)
            self.stream.flush()

    def log_completion(self, record):
        """Record a completed evaluation to the log.

        :param record: Record of the function evaluation
        """
        xstr = np.array_str(record.params[0], max_line_width=np.inf,
                            precision=5, suppress_small=True)
        self.log("{0}:\t{1}\t{2}\n\t{3}".format(
            self.numeval, record.value, "Feasible", xstr))

    def adjust_step(self):
        """Adjust the sampling radius sigma.

        After succtol successful steps, we cut the sampling radius;
        after failtol failed steps, we double the sampling radius.

        :ivar Fnew: Best function value in new step
        :ivar fbest: Previous best function evaluation
        """
        # Initialize if this is the first adaptive step
        if self.fbest_old is None:
            self.fbest_old = self.fbest
            return

        # Check if we succeeded at significant improvement
        if self.fbest < self.fbest_old - 1e-3*math.fabs(self.fbest_old):
            self.status = max(1, self.status+1)
        else:
            self.status = min(-1, self.status-1)
        self.fbest_old = self.fbest

        # Check if step needs adjusting
        if self.status <= -self.failtol:
            self.status = 0
            self.sigma /= 2
            self.log("Reducing sigma")
        if self.status >= self.succtol:
            self.status = 0
            self.sigma = min(2 * self.sigma, self.sigma_max)
            self.log("Increasing sigma")

    def sample_initial(self):
        """Generate and queue an initial experimental design.
        """
        self.log("=== Restart ===")
        self.fhat.reset()
        self.sigma = self.sigma_max
        self.status = 0
        self.xbest = None
        self.fbest_old = None
        self.fbest = np.inf
        self.fhat.reset()
        start_sample = self.design.generate_points()
        start_sample = np.asarray(self.data.xlow) + start_sample * self.xrange
        # Add extra evaluation points provided by the user
        if self.extra is not None:
            start_sample = np.vstack((start_sample, self.extra))

        start_sample = round_vars(self.data, start_sample)
        for j in range(min(start_sample.shape[0], self.maxeval-self.numeval)):
            proposal = self.propose_eval(start_sample[j, :])
            self.resubmitter.rput(proposal)

        self.search.init(start_sample)

    def sample_adapt(self):
        """Generate and queue samples from the search strategy
        """
        self.adjust_step()
        nsamples = min(self.nsamples, self.maxeval-self.numeval)
        self.search.make_points(self.xbest, self.sigma,
                                self.fhat.evals, self.maxeval, True)
        for _ in range(nsamples):
            proposal = self.propose_eval(np.ravel(self.search.next()))
            self.resubmitter.rput(proposal)

    def start_batch(self):
        """Generate and queue a new batch of points
        """
        if self.sigma < self.sigma_min:
            self.sample_initial()
        else:
            self.sample_adapt()

    def propose_action(self):
        """Propose an action
        """
        if self.numeval == self.maxeval:
            return self.propose_terminate()
        elif self.resubmitter.num_eval_outstanding == 0:
            self.start_batch()
        return self.resubmitter.get()

    def on_complete(self, record):
        """Handle completed function evaluation.

        When a function evaluation is completed we need to ask the constraint
        handler if the function value should be modified which is the case for
        say a penalty method. We also need to print the information to the
        logfile, update the best value found so far and notify the GUI that
        an evaluation has completed.

        :param record: Evaluation record
        """
        self.log_completion(record)
        self.numeval += 1
        record.worker_id = self.worker_id
        record.worker_numeval = self.numeval
        self.fhat.add_point(record.params[0], record.value)
        if record.value < self.fbest:
            self.xbest = record.params[0]
            self.fbest = record.value


class SyncStrategyPenalty(SyncStrategyNoConstraints):
    """Parallel synchronous optimization strategy with non-bound constraints.

    This is an extension of SyncStrategyNoConstraints that also works with
    bound constraints. We currently only allow inequality constraints, since
    the candidate based methods don't work well with equality constraints.
    We also assume that the constraints are cheap to evaluate, i.e., so that
    it is easy to check if a given point is feasible. More strategies that
    can handle expensive constraints will be added.

    We use a penalty method in the sense that we try to minimize:

    .. math::
        f(x) + \\mu \\sum_j (\\max(0, g_j(x))^2

    where :math:`g_j(x) \\leq 0` are cheap inequality constraints. As a
    measure of promising function values we let all infeasible points have
    the value of the feasible candidate point with the worst function value,
    since large penalties makes it impossible to distinguish between feasible
    points.

    When it comes to the value of :math:`\\mu`, just choose a very large value.


    """

    def __init__(self, worker_id, data, response_surface, maxeval, nsamples,
                 exp_design=None, search_procedure=None, extra=None,
                 quiet=False, stream=sys.stdout, penalty=1.0E6):
        """Initialize the optimization strategy.

        :param worker_id: Start ID in a multistart setting
        :param data: Problem parameter data structure
        :param response_surface: Surrogate model object
        :param maxeval: Function evaluation budget
        :param nsamples: Number of simultaneous fevals allowed
        :param exp_design: Experimental design
        :param search_procedure: Search procedure for finding
            points to evaluate
        :param extra: Points to be added to the experimental design
        :param quiet: If True, nothing is printed to the stream
        :param stream: Where progress should be printed, sys.stdout is default
        :param penalty: Penalty for violating constraints
        """
        SyncStrategyNoConstraints.__init__(self,  worker_id, data,
                                           response_surface, maxeval,
                                           nsamples, exp_design,
                                           search_procedure, extra,
                                           quiet, stream)
        self.penalty = penalty

    def penalty_fun(self, xx):
        """Computes the penalty for constraints violation

        :param xx: Points to compute the penalty for
        :return: Penalty for constraint violations
        """
        # Get the constraint violations
        vec = np.array(self.data.eval_ineq_constraints(xx))
        # Now apply the penalty for the constraint violation
        vec[np.where(vec < 0.0)] = 0.0
        vec **= 2
        # Surrogate + penalty
        return self.penalty * np.asmatrix(np.sum(vec, axis=1)).T

    def evals(self, xx):
        """Predict function values

        As a measure of promising function values we let all infeasible points
        have the value of the feasible candidate point with the worst function
        value, since large penalties makes it impossible to distinguish
        between feasible points.

        :param xx: Data points
        :return: Predicted function values
        """
        penalty = self.penalty_fun(xx)
        vals = self.fhat.evals(xx)
        ind = (np.where(penalty <= 0.0)[0]).T
        if ind.shape[0] > 1:
            ind2 = (np.where(penalty > 0.0)[0]).T
            ind3 = np.argmax(np.squeeze(vals[ind]))
            vals[ind2] = vals[ind3]
            return vals
        else:
            return vals + penalty

    def sample_adapt(self):
        """Generate and queue samples from the search strategy"""
        self.adjust_step()
        nsamples = min(self.nsamples, self.maxeval-self.numeval)
        self.search.make_points(self.xbest, self.sigma,
                                self.evals, self.maxeval, True)
        for _ in range(nsamples):
            proposal = self.propose_eval(np.ravel(self.search.next()))
            self.resubmitter.rput(proposal)

    def log_completion(self, record, penalty):
        """Record a completed evaluation to the log.

        :param record: Record of the function evaluation
        :param penalty: Penalty for the given point
        """
        xstr = np.array_str(record.params[0], max_line_width=np.inf,
                            precision=5, suppress_small=True)
        feas = "Feasible"
        if penalty > 0.0:
            feas = "Infeasible"
        self.log("{0}:\t{1}\t{2}\n\t{3}".format(
            self.numeval, record.value, feas, xstr))

    def on_complete(self, record):
        """Handle completed function evaluation.

        When a function evaluation is completed we need to ask the constraint
        handler if the function value should be modified which is the case for
        say a penalty method. We also need to print the information to the
        logfile, update the best value found so far and notify the GUI that
        an evaluation has completed.

        :param record: Evaluation record
        """
        x = np.zeros((1, record.params[0].shape[0]))
        x[0, :] = record.params[0]
        penalty = self.penalty_fun(x)[0, 0]
        self.log_completion(record, penalty)
        self.numeval += 1
        record.worker_id = self.worker_id
        record.worker_numeval = self.numeval
        self.fhat.add_point(record.params[0], record.value)
        # Check if the penalty function is a new best
        if record.value + penalty < self.fbest:
            self.xbest = record.params[0]
            self.fbest = record.value + penalty
