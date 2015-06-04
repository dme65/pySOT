#!/usr/bin/env python
"""
.. module:: surrogate_optimizer
   :synopsis: Optimizer for SRBF
.. moduleauthor:: David Bindel <bindel@cornell.edu>,
                  David Eriksson <dme65@cornell.edu>
"""

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from poap.strategy import MultiStartStrategy
from poap.controller import SimTeamController
from sot_sync import SynchronousSRBFStrategy
from constraint_method import PenaltyMethod
from rs_capped import RSCapped
from rbf_interpolant import RBFInterpolant, phi_cubic, \
    linear_tail, dphi_cubic, dlinear_tail
import sys
import numpy as np

# Find the best solution
def get_best_solution(controller, dim):
    fbest = np.inf * np.ones(1)
    xbest = np.zeros((1, dim))
    for record in controller.fevals:
        if record.value < fbest[record.worker_id]:
            fbest[record.worker_id] = record.value
            xbest[record.worker_id, :] = record.params[0]
    return xbest.flatten(), fbest[0]

def add_strategy(controller, data, maxeval, nsamples,
                 response_surface=None, expdesign=None,
                 search_procedure=None, constraint_handler=None,
                 graphics_handle=None):
    """Add a strategy to the controller.

    Depending on the global variable USE_SYNCHRONOUS, we attach
    either a multistart synchronous strategy or an asynchronous
    strategy to the controller.

    Args:
        controller: Controller used to manage the optimization
        data: Structure describing the problem
        maxeval: Maximum allowed fevals
        nstarts: Number of independent runs in a multistart
        nsamples: Number of simultaneous fevals allowed (synchronous)
    """
    strategies = [SynchronousSRBFStrategy(0, data, response_surface,
                                         maxeval, nsamples, expdesign,
                                         search_procedure, constraint_handler,
                                         graphics_handle)]

    strategy = MultiStartStrategy(controller, strategies)
    controller.strategy = strategy

def delay():
    """Generate a delay (simulated feval time) uniformly in [a,b]"""
    return 0.0

def optimize(nthreads, data, maxeval, nsample, response_surface=None,
             experimental_design=None, search_strategies=None,
             constraint_handler=None, graphics_handle=None):

    """Run an optimization with a simulated time elaps controller.

    Right now, we've hardwired this to use a single start with three
    samples at each time step.  Each sample runs for one simulated second
    plus a Unif(0,1) additional delay.

    Args:
        data: Problem definition structure
    """

    """ Start writing to logfile """
    old_stdout = sys.stdout
    log_file = open("surrogate_optimization.log", "w")
    sys.stdout = log_file

    infeasible = True
    xbest = []
    fbest = np.NaN
    if constraint_handler is None:
        constraint_handler = PenaltyMethod(1.0)
    if response_surface is None:
        response_surface = RSCapped(RBFInterpolant(phi=phi_cubic, P=linear_tail,
                                                   dphi=dphi_cubic, dP=dlinear_tail,
                                                   eta=1e-8, maxp=maxeval))  # Surrogate
    while infeasible is True:
        controller = SimTeamController(data.objfunction, delay, nthreads)
        add_strategy(controller, data, maxeval, nsample, response_surface,
                     experimental_design, search_strategies, constraint_handler,
                     graphics_handle)

        def update_callback(record):
            """Callback to add timestamp on completion of feval."""
            if record.is_done():
                record.t1 = controller.time

        def feval_callback(record):
            """Callback to add timestamp on start of feval."""
            record.t0 = controller.time
            record.add_callback(update_callback)

        controller.add_feval_callback(feval_callback)
        controller.run()
        xbest, fbest = get_best_solution(controller, data.dim)
        if (data.constraints is False) or (constraint_handler.outerLoop is False):
            infeasible = False
        else:
            if np.max(data.eval_ineq_constraints(xbest)) <= 0.0:
                infeasible = False
            else:
                constraint_handler.update_penalty()
                if graphics_handle is not None:
                    graphics_handle.printMessage("Best solution infeasible, restarting\n", "blue")
                print "#CH: Best solution is infeasible, increasing penalty to: " + str(constraint_handler.penalty)
                response_surface.reset()

    # Print the best solution to the logfile
    print("\nBest function value: %f" % fbest)
    print("Best solution: " + np.array2string(xbest).replace('\n', ''))
    # Reset the stream
    sys.stdout = old_stdout
    log_file.close()

    return xbest, fbest

