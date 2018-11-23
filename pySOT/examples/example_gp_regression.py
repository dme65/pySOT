"""
.. module:: example_gp_regression
  :synopsis: Example GP Regression
.. moduleauthor:: David Eriksson <dme65@cornell.edu>
"""

from pySOT.adaptive_sampling import CandidateDYCORS
from pySOT.experimental_design import SymmetricLatinHypercube
from pySOT.strategy import GlobalStrategy
from pySOT.surrogate import GPRegressor
from pySOT.optimization_problems import Ackley

from poap.controller import ThreadController, BasicWorkerThread
import numpy as np
import os.path
import logging

def example_gp_regression():
    if not os.path.exists("./logfiles"):
        os.makedirs("logfiles")
    if os.path.exists("./logfiles/example_gp.log"):
        os.remove("./logfiles/example_gp.log")
    logging.basicConfig(filename="./logfiles/example_gp.log",
                        level=logging.INFO)

    print("\nNumber of threads: 4")
    print("Maximum number of evaluations: 500")
    print("Sampling method: CandidateDYCORS")
    print("Experimental design: Symmetric Latin Hypercube")
    print("Surrogate: Gaussian process regression")

    nthreads = 4
    max_evals = 50

    ackley = Ackley(dim=4)
    print(ackley.info)

    gp = GPRegressor(dim=ackley.dim, maxpts=max_evals)
    dycors = CandidateDYCORS(opt_prob=ackley, max_evals=max_evals, numcand=100*ackley.dim)
    slhd = SymmetricLatinHypercube(dim=ackley.dim, npts=2*(ackley.dim+1))

    # Create a strategy and a controller
    controller = ThreadController()
    controller.strategy = \
        GlobalStrategy(max_evals=max_evals, opt_prob=ackley, asynchronous=True,
                       exp_design=slhd, surrogate=gp, adapt_sampling=dycors,
                       batch_size=nthreads)

    # Launch the threads and give them access to the objective function
    for _ in range(nthreads):
        worker = BasicWorkerThread(controller, ackley.eval)
        controller.launch_worker(worker)

    # Run the optimization strategy
    result = controller.run()

    print('Best value found: {0}'.format(result.value))
    print('Best solution found: {0}\n'.format(
        np.array_str(result.params[0], max_line_width=np.inf,
                     precision=5, suppress_small=True)))


if __name__ == '__main__':
    example_gp_regression()
