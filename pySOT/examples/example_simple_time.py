"""
.. module:: example_simple_time
  :synopsis: Example Simple with time budget
.. moduleauthor:: David Eriksson <dme65@cornell.edu>
"""

from pySOT.adaptive_sampling import CandidateDYCORS
from pySOT.experimental_design import SymmetricLatinHypercube
from pySOT.strategy import SRBFStrategy
from pySOT.surrogate import RBFInterpolant, CubicKernel, LinearTail
from pySOT.optimization_problems import Ackley

from poap.controller import ThreadController, BasicWorkerThread
import numpy as np
import os.path
import logging
import time


def example_simple_time():
    if not os.path.exists("./logfiles"):
        os.makedirs("logfiles")
    if os.path.exists("./logfiles/example_simple_time.log"):
        os.remove("./logfiles/example_simple_time.log")
    logging.basicConfig(filename="./logfiles/example_simple_time.log",
                        level=logging.INFO)

    print("\nNumber of threads: 4")
    print("Time budget: 10 seconds")
    print("Sampling method: CandidateDYCORS")
    print("Experimental design: Symmetric Latin Hypercube")
    print("Surrogate: Cubic RBF")

    nthreads = 4
    max_evals = -10  # This means 10 seconds

    ackley = Ackley(dim=30)
    print(ackley.info)

    rbf = RBFInterpolant(dim=ackley.dim, kernel=CubicKernel(),
                         tail=LinearTail(ackley.dim), maxpts=1000)
    dycors = CandidateDYCORS(opt_prob=ackley, max_evals=1000, numcand=100*ackley.dim)
    slhd = SymmetricLatinHypercube(dim=ackley.dim, npts=2*(ackley.dim+1))

    # Create a strategy and a controller
    controller = ThreadController()
    controller.strategy = \
        SRBFStrategy(max_evals=max_evals, opt_prob=ackley, asynchronous=True,
                     exp_design=slhd, surrogate=rbf, adapt_sampling=dycors,
                     batch_size=nthreads)

    # Launch the threads and give them access to the objective function
    for _ in range(nthreads):
        worker = BasicWorkerThread(controller, ackley.eval)
        controller.launch_worker(worker)

    # Run the optimization strategy
    start_time = time.time()
    result = controller.run()
    end_time = time.time()

    print('Run time: {0} seconds'.format(end_time - start_time))
    print('Number of evaluations: {0}'.format(len(controller.fevals)))
    print('Best value found: {0}'.format(result.value))
    print('Best solution found: {0}\n'.format(
        np.array_str(result.params[0], max_line_width=np.inf,
                     precision=5, suppress_small=True)))


if __name__ == '__main__':
    example_simple_time()
