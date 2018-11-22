"""
.. module:: test_mars
  :synopsis: Test MARS
.. moduleauthor:: David Eriksson <dme65@cornell.edu>
"""

from pySOT.adaptive_sampling import CandidateDYCORS
from pySOT.experimental_design import SymmetricLatinHypercube
from pySOT.strategy import SRBFStrategy
from pySOT.optimization_problems import Ackley

from poap.controller import ThreadController, BasicWorkerThread
import numpy as np
import os.path
import logging

# Try to import MARS
try:
    from pySOT.surrogate import MARSInterpolant
except Exception as err:
    print("\nERROR: Failed to import MARS. This is likely "
          "because py-earth is not installed. Aborting.....\n")
    exit()


def test_mars():
    if not os.path.exists("./logfiles"):
        os.makedirs("logfiles")
    if os.path.exists("./logfiles/test_mars.log"):
        os.remove("./logfiles/test_mars.log")
    logging.basicConfig(filename="./logfiles/test_mars.log",
                        level=logging.INFO)

    print("\nNumber of threads: 4")
    print("Maximum number of evaluations: 200")
    print("Sampling method: CandidateDYCORS")
    print("Experimental design: Symmetric Latin Hypercube")
    print("Surrogate: MARS interpolant")

    nthreads = 4
    maxeval = 200
    nsamples = nthreads

    opt_prob = Ackley(dim=5)
    print(opt_prob.info)

    # Create a strategy and a controller
    controller = ThreadController()
    controller.strategy = \
    controller.strategy = \
        SRBFStrategy(worker_id=0, maxeval=maxeval, opt_prob=opt_prob,
                     exp_design=SymmetricLatinHypercube(dim=opt_prob.dim, npts=2 * (opt_prob.dim + 1)),
                     surrogate=MARSInterpolant(opt_prob.dim, maxpts=maxeval),
                     sampling_method=CandidateDYCORS(data=opt_prob, numcand=100*opt_prob.dim),
                     batch_size=nsamples, asynchronous=True)

    # Launch the threads and give them access to the objective function
    for _ in range(nthreads):
        worker = BasicWorkerThread(controller, opt_prob.eval)
        controller.launch_worker(worker)

    # Run the optimization strategy
    result = controller.run()

    print('Best value found: {0}'.format(result.value))
    print('Best solution found: {0}\n'.format(
        np.array_str(result.params[0], max_line_width=np.inf,
                     precision=5, suppress_small=True)))


if __name__ == '__main__':
    test_mars()
