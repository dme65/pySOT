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
max_evals = 200
ackley = Ackley(dim=10)
print(ackley.info)

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

    rbf = RBFInterpolant(dim=ackley.dim, kernel=CubicKernel(),
                         tail=LinearTail(ackley.dim), maxpts=max_evals)
    dycors = CandidateDYCORS(max_evals=max_evals, opt_prob=ackley, numcand=100*ackley.dim)
    slhd = SymmetricLatinHypercube(dim=ackley.dim, npts=2*(ackley.dim+1))

    # Create a strategy and a controller
    controller = ThreadController()
    controller.strategy = \
        SRBFStrategy(max_evals=max_evals, opt_prob=ackley, exp_design=slhd,
                     surrogate=rbf, adapt_sampling=dycors, 
                     asynchronous=True, batch_size=nthreads)

    # Launch the threads and give them access to the objective function
    for _ in range(nthreads):
        worker = BasicWorkerThread(controller, ackley.eval)
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
        worker = BasicWorkerThread(controller, ackley.eval)
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
