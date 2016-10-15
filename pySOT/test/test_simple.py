"""
.. module:: test_simple
  :synopsis: Test Simple
.. moduleauthor:: David Eriksson <dme65@cornell.edu>
"""

from pySOT import *
from poap.controller import ThreadController, BasicWorkerThread
import numpy as np
import os.path


def main():
    if not os.path.exists("./logfiles"):
        os.makedirs("logfiles")
    if os.path.exists("./logfiles/test_simple.log"):
        os.remove("./logfiles/test_simple.log")
    logging.basicConfig(filename="./logfiles/test_simple.log",
                        level=logging.INFO)

    print("\nNumber of threads: 4")
    print("Maximum number of evaluations: 500")
    print("Sampling method: CandidateDYCORS, with weight 0.5")
    print("Experimental design: Symmetric Latin Hypercube")
    print("Surrogate: Cubic RBF, domain scaled to unit box")

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
            response_surface=RBFInterpolant(kernel=CubicKernel, tail=LinearTail,
                                            maxp=maxeval, dim=data.dim),
            sampling_method=CandidateDYCORS(data=data, numcand=100*data.dim, weights=[0.5]))

    # Launch the threads and give them access to the objective function
    for _ in range(nthreads):
        worker = BasicWorkerThread(controller, data.objfunction)
        controller.launch_worker(worker)

    # Run the optimization strategy
    result = controller.run()

    print('Best value found: {0}'.format(result.value))
    print('Best solution found: {0}\n'.format(
        np.array_str(result.params[0], max_line_width=np.inf,
                     precision=5, suppress_small=True)))

if __name__ == '__main__':
    main()
