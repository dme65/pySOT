"""
.. module:: test_simple_time
  :synopsis: Test Simple with time budget
.. moduleauthor:: David Eriksson <dme65@cornell.edu>
"""

from pySOT import Ackley, SyncStrategyNoConstraints, \
    SymmetricLatinHypercube, RBFInterpolant, CubicKernel, \
    LinearTail, CandidateDYCORS
from poap.controller import ThreadController, BasicWorkerThread
import numpy as np
import os.path
import logging
import time


def main():
    if not os.path.exists("./logfiles"):
        os.makedirs("logfiles")
    if os.path.exists("./logfiles/test_simple_time.log"):
        os.remove("./logfiles/test_simple_time.log")
    logging.basicConfig(filename="./logfiles/test_simple_time.log",
                        level=logging.INFO)

    print("\nNumber of threads: 4")
    print("Time budget: 30 seconds")
    print("Sampling method: CandidateDYCORS")
    print("Experimental design: Symmetric Latin Hypercube")
    print("Surrogate: Cubic RBF")

    nthreads = 4
    maxeval = -30
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
            response_surface=RBFInterpolant(kernel=CubicKernel, tail=LinearTail,
                                            maxp=1000),
            sampling_method=CandidateDYCORS(data=data, numcand=100*data.dim))

    # Launch the threads and give them access to the objective function
    for _ in range(nthreads):
        worker = BasicWorkerThread(controller, data.objfunction)
        controller.launch_worker(worker)

    # Run the optimization strategy
    start_time = time.time()
    result = controller.run()
    end_time = time.time()

    print('Run time: {0} seconds'.format(end_time - start_time))
    print('Best value found: {0}'.format(result.value))
    print('Best solution found: {0}\n'.format(
        np.array_str(result.params[0], max_line_width=np.inf,
                     precision=5, suppress_small=True)))

if __name__ == '__main__':
    main()
