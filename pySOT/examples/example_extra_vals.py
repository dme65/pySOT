"""
.. module:: example_extra_vals
  :synopsis: Example extra vals
.. moduleauthor:: David Eriksson <dme65@cornell.edu>
"""

from pySOT.experimental_design import SymmetricLatinHypercube
from pySOT.strategy import SRBFStrategy
from pySOT.surrogate import RBFInterpolant, CubicKernel, LinearTail
from pySOT.optimization_problems import Ackley

from poap.controller import ThreadController, BasicWorkerThread, EvalRecord
import numpy as np
import os.path
import logging


def example_extra_vals():
    if not os.path.exists("./logfiles"):
        os.makedirs("logfiles")
    if os.path.exists("./logfiles/example_extra_vals.log"):
        os.remove("./logfiles/example_extra_vals.log")
    logging.basicConfig(filename="./logfiles/example_extra_vals.log",
                        level=logging.INFO)

    num_threads = 4
    max_evals = 500

    ackley = Ackley(dim=10)
    num_extra = 10
    extra = np.random.uniform(ackley.lb, ackley.ub, (num_extra, ackley.dim))
    extra_vals = np.nan * np.ones((num_extra, 1))
    for i in range(num_extra):  # Evaluate every second point
        if i % 2 == 0:
            extra_vals[i] = ackley.eval(extra[i, :])

    rbf = RBFInterpolant(dim=ackley.dim, kernel=CubicKernel(),
                         tail=LinearTail(ackley.dim))
    slhd = SymmetricLatinHypercube(dim=ackley.dim, num_pts=2*(ackley.dim+1))

    # Create a strategy and a controller
    controller = ThreadController()
    controller.strategy = SRBFStrategy(
        max_evals=max_evals, opt_prob=ackley, exp_design=slhd,
        surrogate=rbf, asynchronous=True, batch_size=num_threads,
        extra_points=extra, extra_vals=extra_vals)

    print("Number of threads: {}".format(num_threads))
    print("Maximum number of evaluations: {}".format(max_evals))
    print("Strategy: {}".format(controller.strategy.__class__.__name__))
    print("Experimental design: {}".format(slhd.__class__.__name__))
    print("Surrogate: {}".format(rbf.__class__.__name__))

    # Append the known function values to the POAP database since
    # POAP won't evaluate these points
    for i in range(len(extra_vals)):
        if not np.isnan(extra_vals[i]):
            record = EvalRecord(
                params=(np.ravel(extra[i, :]),), status='completed')
            record.value = extra_vals[i]
            record.feasible = True
            controller.fevals.append(record)

    # Launch the threads and give them access to the objective function
    for _ in range(num_threads):
        worker = BasicWorkerThread(controller, ackley.eval)
        controller.launch_worker(worker)

    # Run the optimization strategy
    result = controller.run()

    print('Best value found: {0}'.format(result.value))
    print('Best solution found: {0}\n'.format(
        np.array_str(result.params[0], max_line_width=np.inf,
                     precision=5, suppress_small=True)))


if __name__ == '__main__':
    example_extra_vals()
