"""
.. module:: sot_sync
   :synopsis: Parallel synchronous optimization strategy
.. moduleauthor:: David Bindel <bindel@cornell.edu>,
    David Eriksson <dme65@cornell.edu>

:Module: sot_sync
:Author: David Bindel <bindel@cornell.edu>,
    David Eriksson <dme65@cornell.edu>
"""

import sys
import numpy as np
import math
from experimental_design import SymmetricLatinHypercube
from search_procedure import round_vars, CandidateDyCORS
from poap.strategy import Proposal, RetryStrategy

feasvec = ["Infeasible", "Feasible"]


class SynchronousStrategy(object):
    """Parallel synchronous optimization strategy.

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

    def __init__(self, worker_id, data, fhat, maxeval, nsamples,
                 exp_design=None, search_procedure=None,
                 constraint_handler=None):
        """Initialize the optimization strategy.

        :param worker_id: Start ID in a multistart setting
        :param data: Problem parameter data structure
        :param fhat: Surrogate model object
        :param maxeval: Function evaluation budget
        :param nsamples: Number of simultaneous fevals allowed
        :param exp_design: Experimental design
        :param search_procedure: Search procedure for finding
            points to evaluate
        :param constraint_handler: Method for handling non-linear constraints
        """

        self.worker_id = worker_id
        self.data = data
        self.fhat = fhat
        self.maxeval = maxeval
        self.nsamples = nsamples

        # Default to generate sampling points using Symmetric Latin Hypercube
        self.design = exp_design
        if self.design is None:
            self.design = SymmetricLatinHypercube(data.dim, 2*data.dim+1)

        self.xrange = np.asarray(data.xup - data.xlow)

        # algorithm parameters
        self.sigma_max = 0.2  	# w.r.t. unit box
        self.sigma_min = 0.005  # w.r.t. unit box
        self.failtol = max(5, data.dim)
        self.succtol = 3

        self.numeval = 0
        self.status = 0
        self.sigma = 0
        self.resubmitter = RetryStrategy(self)
        self.xbest = None
        self.fbest = np.inf
        self.fbest_old = None

        self.constraint_handler = constraint_handler

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
        print(message)
        sys.stdout.flush()

    def log_completion(self, record):
        """Record a completed evaluation to the log.

        :param record: Record of the function evaluation
        """
        xstr = np.array2string(record.params[0]).replace('\n', '')
        if self.data.constraints and self.constraint_handler is not None:
            isfeas = feasvec[self.constraint_handler.feasible(
                self.data, record.params[0])]
        else:
            isfeas = "Feasible"
        self.log("{0}:\t{1}\t{2}\n\t{3}".format(self.numeval, record.value,
                                                isfeas, xstr))

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
        self.fhat.reset()  # FIXME, evaluations should be saved
        self.sigma = self.sigma_max
        self.status = 0
        self.fbest_old = None
        self.fbest = np.inf
        self.fhat.reset()
        start_sample = self.design.generate_points()
        start_sample = np.asarray(self.data.xlow) + start_sample * self.xrange
        # FIXME, rounding should be moved to experimental design
        start_sample = round_vars(self.data, start_sample)
        for j in range(min(start_sample.shape[0], self.maxeval-self.numeval)):
            self.resubmitter.append(start_sample[j, :])
        self.search.init(self.fhat, start_sample)

    def sample_adapt(self):
        """Generate and queue samples from the search strategy
        """
        self.adjust_step()
        nsamples = min(self.nsamples, self.maxeval-self.numeval)
        self.search.make_points(self.xbest, self.sigma, self.maxeval, True)
        for _ in range(nsamples):
            self.resubmitter.append(np.ravel(self.search.next()))

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
            return Proposal('terminate')
        elif self.resubmitter.num_outstanding() == 0:
            self.start_batch()
        return self.resubmitter.propose_action()

    def on_complete(self, record):
        """Handle completed function evaluation.

        When a function evaluation is completed we need to ask the constraint
        handler if the function value should be modified which is the case for
        say a penalty method. We also need to print the information to the
        logfile, update the best value found so far and notify the GUI that
        an evaluation has completed.

        :param record: Evaluation record
        """
        log_complete(self, record)
        
        # Give the constraint handler a chance to update the value
        if self.data.constraints and self.constraint_handler is not None:
            record.value = self.constraint_handler.eval(self.data,
                                                        record.params[0],
                                                        record.value)
        self.numeval += 1
        record.worker_id = self.worker_id
        record.worker_numeval = self.numeval
        self.fhat.add_point(record.params[0], record.value)
        if record.value < self.fbest:
            self.xbest = record.params[0]
            self.fbest = record.value
