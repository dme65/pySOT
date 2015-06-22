"""
.. module:: test_mixed_integer_constraints
  :synopsis: Test Mixed integer problem with constraints
.. moduleauthor:: David Eriksson <dme65@cornell.edu>
"""

from pySOT import *
from poap.controller import ThreadController, BasicWorkerThread
import numpy as np


def main():
    print("Number of threads: 4")
    print("Maximum number of evaluations: 200")
    print("Search strategy: CandidateDyCORS, CandidateDyCORS_INT"
          ", CandidateDyCORS_CONT, CandidateUniform")
    print("Experimental design: Symmetric Latin Hypercube")
    print("Surrogate: Cubic RBF")

    nthreads = 4
    maxeval = 200
    nsamples = nthreads

    data = LinearMI()
    print(data.info)

    def feasible_merit(record):
        "Merit function for ordering final answers -- kill infeasible x"
        x = record.params[0].reshape((1, record.params[0].shape[0]))
        if np.max(data.eval_ineq_constraints(x)) > 0:
            return np.inf
        return record.value

    exp_design = SymmetricLatinHypercube(dim=data.dim, npts=2*data.dim+1)
    response_surface = RBFInterpolant(phi=phi_cubic, P=linear_tail,
                                      dphi=dphi_cubic, dP=dlinear_tail,
                                      eta=1e-8, maxp=maxeval)

    # Use a multi-search strategy for candidate points
    search_proc = MultiSearchStrategy(
        [CandidateDyCORS(data=data, numcand=200*data.dim),
         CandidateUniform(data=data, numcand=200*data.dim),
         CandidateDyCORS_INT(data=data, numcand=200*data.dim),
         CandidateDyCORS_CONT(data=data, numcand=200*data.dim)],
        [0, 1, 2, 3])

    # Create a strategy and a controller
    controller = ThreadController()
    controller.strategy = \
        SyncStrategyPenalty(
            worker_id=0, data=data,
            response_surface=response_surface,
            maxeval=maxeval, nsamples=nsamples,
            exp_design=exp_design,
            search_procedure=search_proc,
            quiet=True)

    # Launch the threads
    for _ in range(nthreads):
        worker = BasicWorkerThread(controller, data.objfunction)
        controller.launch_worker(worker)

    result = controller.run(merit=feasible_merit)
    best, xbest = result.value, result.params[0]

    print('Best value: {0}'.format(best))
    print('Best solution: {0}'.format(
        np.array_str(xbest, max_line_width=np.inf,
                     precision=5, suppress_small=True)))


if __name__ == '__main__':
    main()
