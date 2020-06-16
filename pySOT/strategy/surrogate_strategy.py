import abc
import logging
import os

import dill
import numpy as np
from poap.strategy import BaseStrategy, Proposal

from ..experimental_design import ExperimentalDesign
from ..optimization_problems import OptimizationProblem
from ..surrogate import Surrogate
from ..utils import check_opt_prob

# Get module-level logger
logger = logging.getLogger(__name__)


class SurrogateBaseStrategy(BaseStrategy):
    __metaclass__ = abc.ABCMeta
    """Surrogate base strategy.

    This is a base strategy for surrogate optimization. This class is abstract
    and inheriting classes must implement generate_evals(self, num_pts)
    which proposes num_pts new evaluations. This strategy follows the general
    idea of surrogate optimization:

    1.  Generate an initial experimental design
    2.  Evaluate the points in the experimental design
    3.  Build a Surrogate model from the data
    4.  Repeat until stopping criterion met
    5.      Generate num_pts new points to evaluate
    6.      Evaluate the new point(s)
    7.      Update the surrogate model

    The optimization problem, experimental design, and surrogate model
    must implement an abstract class definition to make sure they are
    compatible with the framework. More information about required
    methods and attributes is explained in the pySOT documentation:
        https://pysot.readthedocs.io/

    We support function evaluations in serial, synchronous parallel, and
    asynchronous parallel. Serial evaluations can be achieved by using the
    SerialController in POAP and either run this strategy asynchronously
    or synchronously with batch_size equal to 1. The main difference between
    asynchronous and synchronous parallel is that we launch new function
    evaluations as soon as a worker becomes available when using asynchronous
    parallel, while we wait for the entire batch to finish when using
    synchronous parallel. It is important that the experimental design
    generates enough points to construct an initial surrogate model.

    The user can supply additional points to add to the experimental design
    using the extra_points and extra_vals inputs. This is of interest if
    a good starting point is known. The surrogate model is reset by default
    to make sure no old points are present in the surrogate model, but this
    behavior can be modified using the reset_surrogate argument.

    The strategy stops proposing new evaluations once the attribute
    self.terminate has been set to True or when the evaluation budget has been
    exceeded. We wait for pending evaluations to finish before the strategy
    terminates.

    :param max_evals: Evaluation budget
    :type max_evals: int
    :param opt_prob: Optimization problem object
    :type opt_prob: OptimizationProblem
    :param exp_design: Experimental design object
    :type exp_design: ExperimentalDesign
    :param surrogate: Surrogate object
    :type surrogate: Surrogate
    :param asynchronous: Whether or not to use asynchrony (True/False)
    :type asynchronous: bool
    :param batch_size: Size of batch (use 1 for serial, ignored if async)
    :type batch_size: int
    :param extra_points: Extra points to add to the experimental design
    :type extra_points: numpy.array of size n x dim
    :param extra_vals: Values for extra_points (np.nan/np.inf if unknown)
    :type extra_vals: numpy.array of size n x 1
    :param use_restarts: Whether or not to restart after convergence
    :type use_restarts: bool
    """

    def __init__(
        self,
        max_evals,
        opt_prob,
        exp_design,
        surrogate,
        asynchronous=True,
        batch_size=None,
        extra_points=None,
        extra_vals=None,
        use_restarts=True,
    ):
        self.asynchronous = asynchronous
        self.batch_size = batch_size

        # Save the objects
        self.opt_prob = opt_prob
        self.exp_design = exp_design
        self.surrogate = surrogate

        # Sampler state
        self.use_restarts = use_restarts  # Whether we are using restarts
        self.terminate = False  # Termination criterion (eval count by default) reached
        self.converged = False  # Current optimization has converged, we restart if possible
        self.proposal_counter = 0
        self.accepted_count = 0
        self.rejected_count = 0

        # Initial design info
        self.extra_points = extra_points
        self.extra_vals = extra_vals
        self.batch_queue = []  # Unassigned points in initial experiment
        self.init_pending = 0  # Number of outstanding initial fevals
        self.phase = 1  # 1 for initial, 2 for adaptive

        # Budgeting state
        self.num_evals = 0  # Number of completed fevals
        self.max_evals = max_evals  # Remaining feval budget
        self.pending_evals = 0  # Number of outstanding fevals

        # Completed evaluations
        self.X = np.empty([0, opt_prob.dim])
        self.fX = np.empty([0, 1])
        self.Xpend = np.empty([0, opt_prob.dim])
        self.fevals = []

        # Completed evaluations in the current run
        self._X = np.empty([0, opt_prob.dim])
        self._fX = np.empty([0, 1])

        # Event indices to keep track of if points where proposed before an event (restart, parameter change, etc.)
        self.ev_last = 0
        self.ev_next = 1

        # Check inputs (implemented by each strategy)
        self.check_input()

        # Start with first experimental design
        self.sample_initial()

    def get_ev(self):
        """Return event index and increase the counter."""
        ev = self.ev_next
        self.ev_next += 1
        return ev

    @abc.abstractmethod  # pragma: no cover
    def generate_evals(self, num_pts):
        pass

    def check_input(self):
        """Check the inputs to the optimization strategt. """
        if not isinstance(self.surrogate, Surrogate):
            raise ValueError("surrogate must implement Surrogate")
        if not isinstance(self.exp_design, ExperimentalDesign):
            raise ValueError("exp_design must implement ExperimentalDesign")
        if not isinstance(self.opt_prob, OptimizationProblem):
            raise ValueError("opt_prob must implement OptimizationProblem")
        check_opt_prob(self.opt_prob)
        if not self.asynchronous and self.batch_size is None:
            raise ValueError("You must specify batch size in synchronous mode " "(use 1 for serial)")
        if not isinstance(self.max_evals, int) and self.max_evals > 0 and self.max_evals >= self.exp_design.num_pts:
            raise ValueError("max_evals must be an integer >= exp_des.num_pts")

    def save(self, fname):
        """Save the state of the strategy.

        We do this in a 3-step procedure
            1) Save to temp file
            2) Move temp file to save file
            3) Remove temp file

        :param fname: Filename
        :type fname: string
        """
        temp_fname = "temp_" + fname
        with open(temp_fname, "wb") as output:
            dill.dump(self, output, dill.HIGHEST_PROTOCOL)
        os.rename(temp_fname, fname)

    def resume(self):
        """Resume a terminated run."""
        if self.phase == 1:  # Put the points back in the queue in init phase
            self.init_pending = 0
            for x in self.Xpend:
                self.batch_queue.append(np.copy(x))

        # Remove everything that was pending
        self.pending_evals = 0
        self.Xpend = np.empty([0, self.opt_prob.dim])

    def check_termination(self):
        """Check if evaluation based termination criterion has been reached."""
        if self.num_evals + self.pending_evals >= self.max_evals:  # Budget exhausted
            self.terminate = True

    def sample_restart(self):
        """Restart a run after convergence."""
        self.ev_last = self.get_ev()

        # Reset data in current run
        self._X = np.empty([0, self.opt_prob.dim])
        self._fX = np.empty([0, 1])

        # Reset flags
        self.converged = False
        self.phase = 1

        # Generate new initial design
        self.sample_initial()

    def log_completion(self, record):
        """Record a completed evaluation to the log.

        :param record: Record of the function evaluation
        :type record: EvalRecord
        """
        xstr = np.array_str(record.params[0], max_line_width=np.inf, precision=5, suppress_small=True)
        logger.info("{} {:.3e} @ {}".format(self.num_evals, record.value, xstr))

    def sample_initial(self):
        """Generate and queue an initial experimental design."""
        logger.info("=== Start ===")

        # Reset surrogate model
        self.surrogate.reset()

        # NB: Experimental designs can now handle the mapping
        start_sample = self.exp_design.generate_points(
            lb=self.opt_prob.lb, ub=self.opt_prob.ub, int_var=self.opt_prob.int_var
        )
        assert start_sample.shape[1] == self.opt_prob.dim, "Dimension mismatch between problem and experimental design"

        for j in range(self.exp_design.num_pts):
            self.batch_queue.append(start_sample[j, :])

        # We only evaluate these points before the first restart
        if self.extra_points is not None and len(self.X) == 0:
            for i in range(self.extra_points.shape[0]):
                if self.extra_vals is None or np.all(np.isnan(self.extra_vals[i])):  # Unknown value
                    self.batch_queue.append(self.extra_points[i, :])
                else:  # Known value, save point and add to surrogate model
                    x = np.copy(self.extra_points[i, :])
                    self.X = np.vstack((self.X, x))
                    self.fX = np.vstack((self.fX, self.extra_vals[i]))
                    self.surrogate.add_points(x, self.extra_vals[i])

    def propose_action(self):
        """Propose an action.

        We pop points from the initial batch queue if we are still in phase 1, otherwise we make proposals
        in the adaptive phase. We check two flags:
            self.converged: We restart if restarts are enabled
            self.terminate: We terminate the optimization procedure if no evaluations are pending

        NB: We allow workers to continue to the adaptive phase if
        the initial queue is empty. This implies that we need enough
        points in the experimental design for us to construct a
        surrogate.
        """

        # Check if termination criterion has been reached
        self.check_termination()

        # Decide the next action to take
        if self.terminate:  # Check if termination has been triggered
            if self.pending_evals == 0:  # Only terminate if nothing is pending, otherwise take no action
                return Proposal("terminate")
        elif self.converged:
            if self.use_restarts:  # Start a new run
                if self.asynchronous or self.pending_evals == 0:  # We can restart immidiately, else wait
                    self.sample_restart()  # Trigger the restart
                    return self.init_proposal()  # We are now in phase 1, so make an initial proposal
                else:
                    return
        elif self.batch_queue:  # Propose point from the batch_queue
            if self.phase == 1:
                return self.init_proposal()
            else:
                return self.adapt_proposal()
        else:  # Make new proposal in the adaptive phase
            self.phase = 2
            if self.asynchronous:  # Always make proposal with asynchrony
                self.generate_evals(num_pts=1)
            elif self.pending_evals == 0:  # Make sure the entire batch is done
                self.generate_evals(num_pts=self.batch_size)

            # We allow generate_evals to trigger a restart, so check if status has changed
            if self.converged:
                if self.use_restarts:  # Start a new run
                    if self.asynchronous or self.pending_evals == 0:  # We can restart immidiately, else wait
                        self.sample_restart()  # Trigger the restart
                        return self.init_proposal()  # We are now in phase 1, so make an initial proposal
                    else:
                        return

            # Launch the new evaluations (the others will be triggered later)
            return self.adapt_proposal()

    def make_proposal(self, x):
        """Create proposal and update counters and budgets."""
        proposal = Proposal("eval", x)
        self.pending_evals += 1
        self.Xpend = np.vstack((self.Xpend, np.copy(x)))
        return proposal

    def remove_pending(self, x):
        """Delete a pending point from self.Xpend."""
        idx = np.where((self.Xpend == x).all(axis=1))[0]
        self.Xpend = np.delete(self.Xpend, idx, axis=0)

    # == Processing in initial phase ==

    def init_proposal(self):
        """Propose a point from the initial experimental design."""
        proposal = self.make_proposal(self.batch_queue.pop())
        proposal.add_callback(self.on_initial_proposal)
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
        proposal.record.add_callback(self.on_initial_update)
        proposal.record.ev_id = self.get_ev()

    def on_initial_rejected(self, proposal):
        """Handle proposal rejection from initial design."""
        self.rejected_count += 1
        self.pending_evals -= 1
        xx = proposal.args[0]
        self.batch_queue.append(xx)  # Add back to queue
        self.Xpend = np.vstack((self.Xpend, np.copy(xx)))
        self.remove_pending(xx)

    def on_initial_update(self, record):
        """Handle update of feval from initial design."""
        if record.status == "completed":
            self.on_initial_completed(record)
        elif record.is_done:
            self.on_initial_aborted(record)

    def on_initial_completed(self, record):
        """Handle successful completion of feval from initial design."""
        self.num_evals += 1
        self.pending_evals -= 1

        xx, fx = np.copy(record.params[0]), record.value
        self.X = np.vstack((self.X, np.atleast_2d(xx)))
        self.fX = np.vstack((self.fX, fx))
        self.remove_pending(xx)

        # Only count point if it was proposed after the last restart
        if record.ev_id > self.ev_last:
            self._X = np.vstack((self._X, np.atleast_2d(xx)))
            self._fX = np.vstack((self._fX, fx))
            self.surrogate.add_points(xx, fx)

        self.log_completion(record)
        self.fevals.append(record)

    def on_initial_aborted(self, record):
        """Handle aborted feval from initial design."""
        self.pending_evals -= 1
        xx = record.params[0]
        self.batch_queue.append(xx)
        self.remove_pending(xx)
        self.fevals.append(record)

    # == Processing in adaptive phase ==

    def adapt_proposal(self):
        """Propose a point from the batch_queue."""
        if self.batch_queue:
            proposal = self.make_proposal(self.batch_queue.pop())
            proposal.add_callback(self.on_adapt_proposal)
            return proposal

    def on_adapt_proposal(self, proposal):
        """Handle accept/reject of proposal from sampling phase."""
        if proposal.accepted:
            self.on_adapt_accept(proposal)
        else:
            self.on_adapt_reject(proposal)

    def on_adapt_accept(self, proposal):
        """Handle accepted proposal from sampling phase."""
        self.accepted_count += 1
        proposal.record.add_callback(self.on_adapt_update)
        proposal.record.ev_id = self.get_ev()

    def on_adapt_reject(self, proposal):
        """Handle rejected proposal from sampling phase."""
        self.rejected_count += 1
        self.pending_evals -= 1
        xx = np.copy(proposal.args[0])
        self.remove_pending(xx)
        if not self.asynchronous:  # Add back to the queue in synchronous case
            self.batch_queue.append(xx)

    def on_adapt_update(self, record):
        """Handle update of feval from sampling phase."""
        if record.status == "completed":
            self.on_adapt_completed(record)
        elif record.is_done:
            self.on_adapt_aborted(record)

    def on_adapt_completed(self, record):
        """Handle completion of feval from sampling phase."""
        self.num_evals += 1
        self.pending_evals -= 1

        xx, fx = np.copy(record.params[0]), record.value
        self.X = np.vstack((self.X, np.atleast_2d(xx)))
        self.fX = np.vstack((self.fX, fx))
        self.remove_pending(xx)

        # Only count point if it was proposed after the last restart
        if record.ev_id > self.ev_last:
            self._X = np.vstack((self._X, np.atleast_2d(xx)))
            self._fX = np.vstack((self._fX, fx))
            self.surrogate.add_points(xx, fx)

        self.log_completion(record)
        self.fevals.append(record)

    def on_adapt_aborted(self, record):
        """Handle aborted feval from sampling phase."""
        self.pending_evals -= 1
        xx = np.copy(record.params[0])
        self.remove_pending(xx)
        self.fevals.append(record)
