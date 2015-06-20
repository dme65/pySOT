"""
.. module:: test_mixed_integer_constraints
  :synopsis: Test Mixed integer problem with constraints
.. moduleauthor:: David Eriksson <dme65@cornell.edu>
"""

from pySOT import *
from poap.controller import ThreadController, BasicWorkerThread
from poap.strategy import MultiStartStrategy
import numpy as np

#  FIXME, Remove when POAP handles feasibility
def get_best_feas(controller, data):
    # Extract the best feasible solution
    best = np.inf * np.ones(1)
    xbest = np.zeros((1, data.dim))
    for record in controller.fevals:
        x = np.zeros((1, record.params[0].shape[0]))
        x[0, :] = record.params[0]
        if record.value < best[record.worker_id] and np.max(data.eval_ineq_constraints(x)) <= 0.0:
            best[record.worker_id] = record.value
            xbest[record.worker_id, :] = record.params[0]
    return best[0], xbest

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

    exp_design = SymmetricLatinHypercube(dim=data.dim, npts=2*data.dim+1)
    response_surface = RBFInterpolant(phi=phi_cubic, P=linear_tail,
                                      dphi=dphi_cubic, dP=dlinear_tail,
                                      eta=1e-8, maxp=maxeval)

    # Use a multi-search strategy for candidate points
    sp1 = CandidateDyCORS(data=data, numcand=200*data.dim)
    sp2 = CandidateUniform(data=data, numcand=200*data.dim)
    sp3 = CandidateDyCORS_INT(data=data, numcand=200*data.dim)
    sp4 = CandidateDyCORS_CONT(data=data, numcand=200*data.dim)
    search_proc = MultiSearchStrategy([sp1, sp2, sp3, sp4], [0, 1, 2, 3])

    # Create a strategy and a controller
    controller = ThreadController()
    strategy = [SyncStrategyPenalty(worker_id=0, data=data,
                                    response_surface=response_surface,
                                    maxeval=maxeval, nsamples=nsamples,
                                    exp_design=exp_design,
                                    search_procedure=search_proc,
                                    quiet=True)]
    strategy = MultiStartStrategy(controller, strategy)
    controller.strategy = strategy

    # Launch the threads
    for _ in range(nthreads):
        controller.launch_worker(BasicWorkerThread(controller, data.objfunction))

    controller.run()
    best, xbest = get_best_feas(controller, data)

    print('Best value: ' + str(best))
    print('Best solution: ' + np.array_str(xbest[0, :], max_line_width=np.inf, precision=5, suppress_small=True))

if __name__ == '__main__':
    main()
