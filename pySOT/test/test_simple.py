"""
.. module:: test_ensemble
  :synopsis: Test Ensemble surrogates
.. moduleauthor:: David Eriksson <dme65@cornell.edu>
"""

from pySOT import *
from poap.controller import ThreadController, BasicWorkerThread
import numpy as np


def main():
    print("Number of threads: 4")
    print("Maximum number of evaluations: 1000")
    print("Search strategy: Candidate DyCORS")
    print("Experimental design: Latin Hypercube")
    print("Ensemble surrogates: Cubic RBF")

    nthreads = 4
    maxeval = 1000
    nsamples = nthreads

    data = Ackley(dim=10)
    print(data.info)

    # Create a strategy and a controller
    controller = ThreadController()
    controller.strategy = \
        SyncStrategyNoConstraints(
            worker_id=0, data=data,
            maxeval=maxeval, nsamples=nsamples,
            exp_design=LatinHypercube(dim=data.dim, npts=2*data.dim+1),
            response_surface=RBFInterpolant(phi=phi_cubic, P=linear_tail,
                                            dphi=dphi_cubic, dP=dlinear_tail,
                                            eta=1e-8, maxp=maxeval),
            search_procedure=CandidateDyCORS(data=data, numcand=200*data.dim),
            quiet=True)

    # Launch the threads and give them access to the objective function
    for _ in range(nthreads):
        worker = BasicWorkerThread(controller, data.objfunction)
        controller.launch_worker(worker)

    # Run the optimization strategy
    result = controller.run()

    print('Best value found: {0}'.format(result.value))
    print('Best solution found: {0}'.format(
        np.array_str(result.params[0], max_line_width=np.inf,
                     precision=5, suppress_small=True)))


if __name__ == '__main__':
    main()
