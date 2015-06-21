"""
.. module:: test_ensemble
  :synopsis: Test Ensemble surrogates
.. moduleauthor:: David Eriksson <dme65@cornell.edu>
"""

from pySOT import *
from poap.controller import ThreadController, BasicWorkerThread
from poap.strategy import MultiStartStrategy
import numpy as np

def main():
    print("Number of threads: 4")
    print("Maximum number of evaluations: 50")
    print("Search strategy: Candidate SRBF")
    print("Experimental design: Latin Hypercube + point [0.1, 0.5, 0.8]")
    print("Surrogate: Cubic RBF")

    nthreads = 4
    maxeval = 50
    nsamples = nthreads

    data = Hartman3()
    print(data.info)
    exp_design = LatinHypercube(dim=data.dim, npts=2*data.dim+1)

    # Use 3 differents RBF's and MARS as an ensemble surrogate
    fhat1 = RBFInterpolant(phi_cubic, linear_tail, dphi_cubic, dlinear_tail, 1e-8, maxeval)
    fhat2 = RBFInterpolant(phi_linear, const_tail, dphi_linear, dconst_tail, 1e-8, maxeval)
    fhat3 = RBFInterpolant(dphi_plate, linear_tail, dphi_plate, dlinear_tail, 1e-8, maxeval)
    fhat4 = MARSInterpolant(maxeval)

    models = [fhat1, fhat2, fhat3, fhat4]
    response_surface = EnsembleSurrogate(models, maxeval)
    search_proc = CandidateSRBF(data=data, numcand=200*data.dim)

    # Add an additional point to the experimental design. If a good solution is already known
    # you can add this point to the experimental design
    extra = np.atleast_2d([0.1, 0.5, 0.8])

    # Print all info to the following file (instead of stdout)
    stream = open("./surrogate_optimizer.log", 'w')

    # Create a strategy and a controller
    controller = ThreadController()
    strategy = [SyncStrategyNoConstraints(worker_id=0, data=data,
                                          response_surface=response_surface,
                                          maxeval=maxeval, nsamples=nsamples,
                                          exp_design=exp_design,
                                          search_procedure=search_proc,
                                          stream=stream, extra=extra)]
    strategy = MultiStartStrategy(controller, strategy)
    controller.strategy = strategy

    # Launch the threads and give them access to the objective function
    for _ in range(nthreads):
        controller.launch_worker(BasicWorkerThread(controller, data.objfunction))

    # Run the optimization strategy
    result = controller.run()

    response_surface.compute_weights()
    print('Final weights: ' + np.array_str(response_surface.weights, max_line_width=np.inf,
                                           precision=5, suppress_small=True))

    print('Best value found: ' + str(result.value))
    print('Best solution found: ' + np.array_str(result.params[0], max_line_width=np.inf,
                                                 precision=5, suppress_small=True))

if __name__ == '__main__':
    main()
