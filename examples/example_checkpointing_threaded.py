"""
.. module:: example_checkpointing_threaded
  :synopsis: Example Checkpointing Threaded
.. moduleauthor:: David Eriksson <dme65@cornell.edu>
"""

import multiprocessing
import os
import time

import numpy as np
from poap.controller import BasicWorkerThread, ThreadController

from pySOT.controller import CheckpointController
from pySOT.experimental_design import SymmetricLatinHypercube
from pySOT.optimization_problems import Ackley
from pySOT.strategy import SRBFStrategy
from pySOT.surrogate import CubicKernel, LinearTail, RBFInterpolant

num_threads = 4
max_evals = 200
ackley = Ackley(dim=10)
print(ackley.info)

fname = "checkpoint.pysot"


def example_checkpoint_threaded():
    if os.path.isfile(fname):
        os.remove(fname)

    # Run for 3 seconds and kill the controller
    p = multiprocessing.Process(target=init, args=())
    p.start()
    time.sleep(3)
    p.terminate()
    p.join()

    print("Die controller, die!")

    # Resume the run
    resume()


def init():
    print("\nInitializing run...")

    rbf = RBFInterpolant(dim=ackley.dim, kernel=CubicKernel(), tail=LinearTail(ackley.dim))
    slhd = SymmetricLatinHypercube(dim=ackley.dim, num_pts=2 * (ackley.dim + 1))

    # Create a strategy and a controller
    controller = ThreadController()
    controller.strategy = SRBFStrategy(
        max_evals=max_evals, opt_prob=ackley, exp_design=slhd, surrogate=rbf, asynchronous=True, batch_size=num_threads
    )

    print("Number of threads: {}".format(num_threads))
    print("Maximum number of evaluations: {}".format(max_evals))
    print("Strategy: {}".format(controller.strategy.__class__.__name__))
    print("Experimental design: {}".format(slhd.__class__.__name__))
    print("Surrogate: {}".format(rbf.__class__.__name__))

    # Launch the threads and give them access to the objective function
    for _ in range(num_threads):
        worker = BasicWorkerThread(controller, ackley.eval)
        controller.launch_worker(worker)

    # Wrap controller in checkpoint object
    controller = CheckpointController(controller, fname=fname)
    result = controller.run()
    print("Best value found: {0}".format(result.value))
    print(
        "Best solution found: {0}\n".format(
            np.array_str(result.params[0], max_line_width=np.inf, precision=5, suppress_small=True)
        )
    )


def resume():
    print("Resuming run...\n")
    controller = ThreadController()

    # Launch the threads and give them access to the objective function
    for _ in range(num_threads):
        worker = BasicWorkerThread(controller, ackley.eval)
        controller.launch_worker(worker)

    # Wrap controller in checkpoint object
    controller = CheckpointController(controller, fname=fname)
    result = controller.resume()
    print("Best value found: {0}".format(result.value))
    print(
        "Best solution found: {0}\n".format(
            np.array_str(result.params[0], max_line_width=np.inf, precision=5, suppress_small=True)
        )
    )


if __name__ == "__main__":
    example_checkpoint_threaded()
