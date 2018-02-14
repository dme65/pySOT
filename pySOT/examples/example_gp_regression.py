"""
.. module:: test_gp_regression
  :synopsis: Test GP Regression
.. moduleauthor:: David Eriksson <dme65@cornell.edu>
"""

from pySOT.adaptive_sampling import CandidateDYCORS
from pySOT.experimental_design import SymmetricLatinHypercube
from pySOT.strategy import SyncStrategyNoConstraints
from pySOT.optimization_problems import Ackley

from poap.controller import ThreadController, BasicWorkerThread
import numpy as np
import os.path
import logging

# Try to import MARS
try:
    from pySOT.surrogate import GPRegression
except Exception as err:
    print("\nERROR: Failed to import GPRegression. This is likely "
          "because scikit-learn>=0.18 is not installed. Aborting.....\n")
    exit()


def main():
    if not os.path.exists("./logfiles"):
        os.makedirs("logfiles")
    if os.path.exists("./logfiles/test_gp.log"):
        os.remove("./logfiles/test_gp.log")
    logging.basicConfig(filename="./logfiles/test_gp.log",
                        level=logging.INFO)

    print("\nNumber of threads: 4")
    print("Maximum number of evaluations: 500")
    print("Sampling method: CandidateDYCORS")
    print("Experimental design: Symmetric Latin Hypercube")
    print("Surrogate: Gaussian process regression")

    nthreads = 4
    maxeval = 500
    nsamples = nthreads

    data = Ackley(dim=10)
    print(data.info)

    # Create a strategy and a controller
    controller = ThreadController()
    controller.strategy = \
        SyncStrategyNoConstraints(
            worker_id=0, data=data,
            maxeval=maxeval, nsamples=nsamples,
            exp_design=SymmetricLatinHypercube(dim=data.dim, npts=2*(data.dim+1)),
            response_surface=GPRegression(dim=data.dim, maxpts=maxeval),
            sampling_method=CandidateDYCORS(data=data, numcand=100*data.dim))

    # Launch the threads and give them access to the objective function
    for _ in range(nthreads):
        worker = BasicWorkerThread(controller, data.eval)
        controller.launch_worker(worker)

    # Run the optimization strategy
    result = controller.run()

    print('Best value found: {0}'.format(result.value))
    print('Best solution found: {0}\n'.format(
        np.array_str(result.params[0], max_line_width=np.inf,
                     precision=5, suppress_small=True)))


if __name__ == '__main__':
    main()
