"""
.. module:: test_constraints
  :synopsis: Test constrained optimization strategy
.. moduleauthor:: David Eriksson <dme65@cornell.edu>
"""

from pySOT import *
from poap.controller import ThreadController, BasicWorkerThread
import numpy as np


def feasible_merit(record, data):
    x = record.params[0].reshape((1, record.params[0].shape[0]))
    if np.max(data.eval_ineq_constraints(x) > 0):
        return np.inf
    return record.value


def main():
    print("Number of threads: 4")
    print("Maximum number of evaluations: 1000")
    print("Search strategy: CandidateDycors")
    print("Experimental design: Latin Hypercube")
    print("Surrogate: Cubic RBF")

    nthreads = 4
    maxeval = 1000
    nsamples = nthreads

    data = Keane(dim=30)
    print(data.info)

    # Create a strategy and a controller
    controller = ThreadController()
    controller.strategy = \
        SyncStrategyPenalty(
            worker_id=0, data=data,
            maxeval=maxeval, nsamples=nsamples, quiet=True,
            response_surface=RBFInterpolant(phi=phi_cubic, P=linear_tail,
                                            dphi=dphi_cubic, dP=dlinear_tail,
                                            eta=1e-8, maxp=maxeval),
            exp_design=LatinHypercube(dim=data.dim, npts=2*data.dim+1),
            search_procedure=CandidateDyCORS(data=data, numcand=5000))

    # Launch the threads
    for _ in range(nthreads):
        worker = BasicWorkerThread(controller, data.objfunction)
        controller.launch_worker(worker)

    result = controller.run(merit=lambda r: feasible_merit(r, data))
    best, xbest = result.value, result.params[0]

    print('Best value: {0}'.format(best))
    print('Best solution: {0}'.format(
        np.array_str(xbest, max_line_width=np.inf,
                     precision=5, suppress_small=True)))


if __name__ == '__main__':
    main()
