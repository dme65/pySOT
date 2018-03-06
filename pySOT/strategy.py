"""
.. module:: strategy
   :synopsis: Parallel synchronous optimization strategy

.. moduleauthor:: David Bindel <bindel@cornell.edu>,
                David Eriksson <dme65@cornell.edu>

:Module: sot_sync_strategies
:Author: David Bindel <bindel@cornell.edu>,
        David Eriksson <dme65@cornell.edu>

"""

from __future__ import print_function

import logging
import math
import time
import abc
import six

from poap.strategy import BaseStrategy, Proposal
from pySOT.surrogate import RBFInterpolant, CubicKernel, LinearTail, RSPenalty
from pySOT.adaptive_sampling import CandidateDYCORS
from pySOT.experimental_design import SymmetricLatinHypercube, LatinHypercube
from pySOT.utils import *

# Get module-level logger
logger = logging.getLogger(__name__)


@six.add_metaclass(abc.ABCMeta)
class SurrogateStrategy(BaseStrategy):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def propose_action(self):  # pragma: no cover
        pass

    # @abc.abstractmethod
    # def save_state(self):  # pragma: no cover
    #     pass
    #
    # @abc.abstractmethod
    # def log_event(self):  # pragma: no cover
    #     pass


class SRBFStrategy(SurrogateStrategy):
    """Parallel asynchronous SRBF optimization strategy.

    In the asynchronous version of SRBF, workers are given function
    evaluations to start on as soon as they become available (unless
    the initial experiment design has been assigned but not completed).
    As evaluations are completed, different actions are taken depending
    on how recent they are.  A "fresh" value is one that was assigned
    since the last time the sampling radius was checked; an "old"
    value is one that was assigned before the last check of the sampling
    radius, but since the last restart; and an "ancient" value is one
    that was assigned before the last restart.  Only fresh values are
    used in adjusting the sampling radius.  Fresh or old values are
    used in determing the best point found since restart (used for
    the center point for sampling).  Any value can be incorporated into
    the response surface.  Sample points are chosen based on a merit
    function that depends not only on the response surface and the distance
    from any previous sample points, but also on the distance from any
    pending sample points.

    Once the budget of maxeval function evaluations have been assigned,
    no further evaluations are assigned to processors.  The code returns
    once all evaluations are completed.
    """

    def __init__(self, worker_id, maxeval, opt_prob, stopping_criterion=None, surrogate=None,
                 exp_design=None, sampling_method=None, extra=None, async=True, batch_size=None):
        """Initialize the asynchronous SRBF optimization.

        Args:
            worker_id: ID of current worker/start in a multistart setting
            data: Problem parameter data structure
            surrogate: Surrogate model object
            maxeval: Function evaluation budget
            design: Experimental design

        """

        self.stopping_criterion = stopping_criterion
        self.proposal_counter = 0
        self.terminate = False
        self.async = async
        self.batch_size = batch_size

        self.worker_id = worker_id
        self.opt_prob = opt_prob
        self.surrogate = surrogate
        if self.surrogate is None:
            self.surrogate = RBFInterpolant(opt_prob.dim, kernel=CubicKernel(), tail=LinearTail(opt_prob.dim), maxpts=maxeval)

        self.maxeval = maxeval
        self.extra = extra

        # Default to generate sampling points using Symmetric Latin Hypercube
        self.design = exp_design
        if self.design is None:
            if maxeval > 10*self.opt_prob.dim:
                self.design = SymmetricLatinHypercube(opt_prob.dim, 2*(opt_prob.dim+1))
            else:
                self.design = LatinHypercube(opt_prob.dim, opt_prob.dim + 1 + batch_size)

        self.xrange = np.asarray(opt_prob.ub - opt_prob.lb)

        # algorithm parameters
        self.sigma_min = 0.2 * (0.5 ** 6)
        self.sigma_max = 0.2
        self.sigma_init = self.sigma_max

        # We divide the tolerance by the batch size in synchronous mode since we process one batch at a time
        if self.async:
            self.failtol = int(max(np.ceil(float(opt_prob.dim)), np.ceil(4.0)))
        else:
            self.failtol = int(max(np.ceil(float(opt_prob.dim) / float(batch_size)), np.ceil(4.0 / float(batch_size))))
        self.succtol = 3
        self.maxfailtol = 4 * self.failtol

        # Budgeting state
        self.numeval = 0             # Number of completed fevals
        self.feval_budget = maxeval  # Remaining feval budget
        self.feval_pending = 0       # Number of outstanding fevals

        # Event indices
        self.ev_last = 0     # Last event index
        self.ev_adjust = 0   # Last sampling adjustment
        self.ev_restart = 0  # Last restart

        # Initial design info
        self.batch_queue = []   # Unassigned points in initial experiment
        self.init_pending = 0   # Number of outstanding initial fevals
        self.phase = 1          # 1 for initial, 2 for adaptive

        # Sampler state
        self.sigma = 0         # Sampling radius
        self.status = 0        # Status counter
        self.failcount = 0     # Failure counter
        self.xbest = None      # Current best x
        self.fbest = np.inf    # Current best f
        self.fbest_new = None  # Best f that hasn't been processed
        self.xbest_new = np.inf  # Best x that hasn't been processed
        self.avoid = {}        # Points to avoid

        self.rejected_count = 0
        self.accepted_count = 0

        # Set up search procedures and initialize
        self.search = sampling_method
        if self.search is None:
            self.search = CandidateDYCORS(opt_prob)

        self.check_input()

        # Start with first experimental design
        self.sample_initial()

    def check_input(self):
        pass

    def proj_fun(self, x):
        return round_vars(x, self.opt_prob.int_var, self.opt_prob.lb, self.opt_prob.ub)

    def log_completion(self, record):
        """Record a completed evaluation to the log.

        :param record: Record of the function evaluation
        """
        xstr = np.array_str(record.params[0], max_line_width=np.inf,
                            precision=5, suppress_small=True)
        if record.feasible:
            logger.info("{} {} {:.3e} @ {}".format("True", self.numeval, record.value, xstr))
        else:
            logger.info("{} {} {:.3e} @ {}".format("False", self.numeval, record.value, xstr))

    def get_ev(self):
        """Get event identifier."""
        self.ev_last += 1
        return self.ev_last

    def adjust_step(self):
        """Adjust the sampling radius sigma.

        After succtol successful steps, we cut the sampling radius;
        after failtol failed steps, we double the sampling radius.

        Args:
            Fnew: Best function value in new step
            fbest: Previous best function evaluation
        """

        # Check if we succeeded at significant improvement
        if self.fbest_new < self.fbest - 1e-3*math.fabs(self.fbest):
            self.fbest = self.fbest_new
            self.xbest = np.copy(self.xbest_new)

            self.fbest_new = np.inf
            self.xbest_new = None

            self.status = max(1, self.status + 1)
            self.failcount = 0
        else:
            self.status = min(-1, self.status - 1)
            self.failcount += 1

        # Check if step needs adjusting
        if self.status <= -self.failtol:
            self.ev_adjust = self.get_ev()
            self.status = 0
            self.sigma /= 2
            logger.info("Reducing sigma")
        if self.status >= self.succtol:
            self.ev_adjust = self.get_ev()
            self.status = 0
            self.sigma = min([2.0 * self.sigma, self.sigma_max])
            logger.info("Increasing sigma")

        # Check if we need to restart
        if self.failcount >= self.maxfailtol or self.sigma <= self.sigma_min:
            self.ev_adjust = self.get_ev()
            self.ev_restart = self.get_ev()
            self.sample_initial()

    def sample_initial(self):
        """Generate and queue an initial experimental design."""
        if self.numeval == 0:
            logger.info("=== Start ===")
        else:
            logger.info("=== Restart ===")
        self.sigma = self.sigma_init
        self.status = 0
        self.failcount = 0
        self.fbest_new = np.inf
        self.fbest = np.inf
        self.surrogate.reset()
        self.phase = 1

        start_sample = self.design.generate_points()
        assert start_sample.shape[1] == self.opt_prob.dim, \
            "Dimension mismatch between problem and experimental design"
        start_sample = from_unit_box(start_sample, self.opt_prob.lb, self.opt_prob.ub)
        if self.extra is not None:
            start_sample = np.vstack((start_sample, self.extra))

        self.init_pending = 0
        for j in range(start_sample.shape[0]):
            start_sample[j, :] = self.proj_fun(start_sample[j, :])  # Project onto feasible region
            self.batch_queue.append(start_sample[j, :])

        self.search.init(start_sample, self.surrogate, self.maxeval - self.numeval)

    def propose_action(self):
        """Propose an action.

        NB: We allow workers to continue to the adaptive phase if the initial queue is empty.
        This implies that we need
        enough points in the experimental design for us to construct a surrogate.
        """
        if self.feval_budget == 0 or self.terminate:
            if self.feval_pending == 0:
                return Proposal('terminate')
            return None
        elif self.async:  # In asynchronous mode
            if self.batch_queue:
                return self.init_proposal()
            else:
                return self.adapt_proposal_async()
        else:  # In synchronous mode
            if self.batch_queue:
                if self.phase == 1:
                    return self.init_proposal()
                else:
                    return self.adapt_proposal_sync()
            elif self.feval_pending == 0:  # Nothing to process
                if self.phase == 2:
                    self.adjust_step()
                    if self.phase == 1:  # This happens if we are restarting, so pop from the new queue
                        return self.init_proposal()
                self.phase = 2  # Mark that we are now the adaptive phase
                self.make_batch()
                return self.adapt_proposal_sync()

    def make_proposal(self, x):
        """Create proposal and update counters and budgets."""
        proposal = Proposal('eval', x)
        self.feval_budget -= 1
        self.feval_pending += 1
        proposal.ev_id = self.get_ev()
        self.avoid[proposal.ev_id] = x
        return proposal

    # == Processing in initial phase ==

    def init_proposal(self):
        """Propose a point from the initial experimental design."""
        proposal = self.make_proposal(self.batch_queue.pop())
        proposal.add_callback(self.on_initial_proposal)
        self.init_pending += 1
        return proposal

    def on_initial_proposal(self, proposal):
        """Handle accept/reject of proposal from initial design."""
        if proposal.accepted:
            self.on_initial_accepted(proposal)
        else:
            self.on_initial_rejected(proposal)

    def on_initial_accepted(self, proposal):
        """Handle proposal accept from initial design."""
        self.accepted_count += 1
        proposal.record.sigma = np.nan
        proposal.record.pred_val = np.nan
        proposal.record.min_dist = np.nan
        proposal.record.ev_id = proposal.ev_id
        proposal.record.add_callback(self.on_initial_update)

    def on_initial_rejected(self, proposal):
        """Handle proposal rejection from initial design."""
        self.rejected_count += 1
        self.feval_budget += 1
        self.feval_pending -= 1
        self.init_pending -= 1
        self.batch_queue.append(proposal.args[0])

    def on_initial_update(self, record):
        """Handle update of feval from initial design."""
        if record.status == 'completed':
            self.on_initial_completed(record)
        elif record.is_done:
            self.on_initial_aborted(record)

    def on_initial_completed(self, record):
        """Handle successful completion of feval from initial design."""

        if self.stopping_criterion is not None:
            if self.stopping_criterion(record.value):
                self.terminate = True

        self.numeval += 1
        self.feval_pending -= 1
        self.init_pending -= 1
        self.surrogate.add_points(np.copy(record.params[0]), record.value)
        del self.avoid[record.ev_id]
        record.worker_id = self.worker_id
        record.worker_numeval = self.numeval
        record.feasible = True
        if record.value < self.fbest:
            self.xbest = np.copy(record.params[0])
            self.fbest = record.value
        self.log_completion(record)

    def on_initial_aborted(self, record):
        """Handle aborted feval from initial design."""
        self.feval_budget += 1
        self.feval_pending -= 1
        self.init_pending -= 1
        self.batch_queue.append(record.params)

    # == Processing in adaptive phase ==

    def adapt_proposal_async(self):
        """Generate the next adaptive sample point."""

        self.proposal_counter += 1
        new_point = self.search.make_points(npts=1, xbest=self.xbest, sigma=self.sigma,
                                            proj_fun=self.proj_fun)
        x = np.ravel(np.asarray(new_point))
        proposal = self.make_proposal(x)
        proposal.sigma = self.sigma
        proposal.pred_val = self.surrogate.eval(x)
        proposal.add_callback(self.on_adapt_proposal)
        return proposal

    def adapt_proposal_sync(self):
        """Generate the next adaptive sample point."""

        self.proposal_counter += 1
        x = np.ravel(np.asarray(self.batch_queue.pop()))
        proposal = self.make_proposal(x)
        proposal.sigma = self.sigma
        proposal.pred_val = self.surrogate.eval(x)
        proposal.add_callback(self.on_adapt_proposal)
        return proposal

    def make_batch(self):
        """Generate the next adaptive sample point."""

        nsamples = min(self.batch_size, self.maxeval - self.numeval)
        new_points = self.search.make_points(npts=nsamples, xbest=np.copy(self.xbest), sigma=self.sigma,
                                             proj_fun=self.proj_fun)
        for i in range(nsamples):
            x = np.copy(np.ravel(new_points[i, :]))
            self.batch_queue.append(x)

    def on_adapt_proposal(self, proposal):
        """Handle accept/reject of proposal from sampling phase."""
        if proposal.accepted:
            self.on_adapt_accept(proposal)
        else:
            self.on_adapt_reject(proposal)

    def on_adapt_accept(self, proposal):
        """Handle accepted proposal from sampling phase."""
        self.accepted_count += 1
        proposal.record.ev_id = proposal.ev_id
        proposal.record.sigma = proposal.sigma
        proposal.record.pred_val = proposal.pred_val
        proposal.record.add_callback(self.on_adapt_update)

    def on_adapt_reject(self, proposal):
        """Handle rejected proposal from sampling phase."""
        self.rejected_count += 1
        self.feval_budget += 1
        self.feval_pending -= 1
        self.search.remove_point(self.avoid[proposal.ev_id])
        del self.avoid[proposal.ev_id]

    def on_adapt_update(self, record):
        """Handle update of feval from sampling phase."""
        if record.status == 'completed':
            self.on_adapt_completed(record)
        elif record.is_done:
            self.on_adapt_aborted(record)

    def on_adapt_completed(self, record):
        """Handle completion of feval from sampling phase."""

        if self.stopping_criterion is not None:
            if self.stopping_criterion(record.value):
                self.terminate = True

        self.numeval += 1
        self.feval_pending -= 1
        self.surrogate.add_points(record.params[0], record.value)
        del self.avoid[record.ev_id]
        record.worker_id = self.worker_id
        record.worker_numeval = self.numeval
        record.feasible = True
        if record.ev_id >= self.ev_restart and record.value < self.fbest_new:
            self.xbest_new = np.copy(record.params[0])
            self.fbest_new = record.value
        self.log_completion(record)
        if self.async:  # Adjust the radius immediately if in async mode
            if record.ev_id >= self.ev_adjust:
                self.adjust_step()

    def on_adapt_aborted(self, record):
        """Handle aborted feval from sampling phase."""
        self.feval_budget += 1
        self.feval_pending -= 1
        self.search.remove_point(self.avoid[record.ev_id])
        del self.avoid[record.ev_id]

    def rec_age(self, record):
        """Return whether a completed record is fresh, old, ancient."""
        if record.ev_id >= self.ev_adjust:
            return "Fresh"
        elif record.ev_id >= self.ev_restart:
            return "Old  "
        else:
            return "Ancient"
