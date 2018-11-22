"""
.. module:: test_checkpointing_threaded
  :synopsis: Test Checkpointing Threaded
.. moduleauthor:: David Eriksson <dme65@cornell.edu>
"""

from pySOT.adaptive_sampling import CandidateDYCORS
from pySOT.experimental_design import SymmetricLatinHypercube
from pySOT.strategy import SRBFStrategy
from pySOT.surrogate import RBFInterpolant, CubicKernel, LinearTail
from pySOT.optimization_problems import Ackley
from pySOT.controller import CheckpointController

from poap.controller import ThreadController, BasicWorkerThread
import numpy as np
import multiprocessing
import time
import os

nthreads = 4
maxeval = 200
nsamples = nthreads

opt_prob = Ackley(dim=10)
print(opt_prob.info)
fname = "checkpoint.pysot"


def test_checkpoint_threaded():
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
    surrogate = RBFInterpolant(
        opt_prob.dim, kernel=CubicKernel(), tail=LinearTail(opt_prob.dim))

    # Create a strategy and a controller
    controller = ThreadController()
    controller.strategy = \
        SRBFStrategy(worker_id=0, maxeval=maxeval, opt_prob=opt_prob,
                     exp_design=SymmetricLatinHypercube(dim=opt_prob.dim,
                                                        npts=2 * (opt_prob.dim + 1)),
                     surrogate=surrogate,
                     sampling_method=CandidateDYCORS(data=opt_prob,
                                                     numcand=100*opt_prob.dim),
                     batch_size=nsamples, asynchronous=True)

    # Launch the threads and give them access to the objective function
    for _ in range(nthreads):
        worker = BasicWorkerThread(controller, opt_prob.eval)
        controller.launch_worker(worker)

    # Wrap controller in checkpoint object
    controller = CheckpointController(controller, fname=fname)
    result = controller.run()
    print('Best value found: {0}'.format(result.value))
    print('Best solution found: {0}\n'.format(
        np.array_str(result.params[0], max_line_width=np.inf,
                     precision=5, suppress_small=True)))


def resume():
    print("Resuming run...\n")
    controller = ThreadController()

    # Launch the threads and give them access to the objective function
    for _ in range(nthreads):
        worker = BasicWorkerThread(controller, opt_prob.eval)
        controller.launch_worker(worker)

    # Wrap controller in checkpoint object
    controller = CheckpointController(controller, fname=fname)
    result = controller.resume()
    print('Best value found: {0}'.format(result.value))
    print('Best solution found: {0}\n'.format(
        np.array_str(result.params[0], max_line_width=np.inf,
                     precision=5, suppress_small=True)))


if __name__ == '__main__':
    test_checkpoint_threaded()
