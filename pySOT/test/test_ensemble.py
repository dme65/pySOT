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
    print("Maximum number of evaluations: 50")
    print("Search strategy: Candidate SRBF")
    print("Experimental design: Latin Hypercube + point [0.1, 0.5, 0.8]")
    print("Surrogate: Cubic RBF, Linear RBF, Thin-plate RBF, MARS")

    nthreads = 4
    maxeval = 50
    nsamples = nthreads

    data = Hartman3()
    print(data.info)

    # Use 3 differents RBF's and MARS as an ensemble surrogate
    models = [
        RBFInterpolant(phi_cubic, linear_tail, dphi_cubic,
                       dlinear_tail, 1e-8, maxeval),
        RBFInterpolant(phi_linear, const_tail, dphi_linear,
                       dconst_tail, 1e-8, maxeval),
        RBFInterpolant(dphi_plate, linear_tail, dphi_plate,
                       dlinear_tail, 1e-8, maxeval),
        MARSInterpolant(maxeval)
    ]
    response_surface = EnsembleSurrogate(models, maxeval)

    # Add an additional point to the experimental design. If a good
    # solution is already known you can add this point to the
    # experimental design
    extra = np.atleast_2d([0.1, 0.5, 0.8])

    # Print all info to the following file (instead of stdout)
    stream = open("./surrogate_optimizer.log", 'w')

    # Create a strategy and a controller
    controller = ThreadController()
    controller.strategy = \
        SyncStrategyNoConstraints(
            worker_id=0, data=data,
            response_surface=response_surface,
            maxeval=maxeval, nsamples=nsamples,
            exp_design=LatinHypercube(dim=data.dim, npts=2*data.dim+1),
            search_procedure=CandidateSRBF(data=data, numcand=200*data.dim),
            stream=stream, extra=extra)

    # Launch the threads and give them access to the objective function
    for _ in range(nthreads):
        worker = BasicWorkerThread(controller, data.objfunction)
        controller.launch_worker(worker)

    # Run the optimization strategy
    result = controller.run()

    response_surface.compute_weights()
    print('Final weights: {0}'.format(
        np.array_str(response_surface.weights, max_line_width=np.inf,
                     precision=5, suppress_small=True)))

    print('Best value found: {0}'.format(result.value))
    print('Best solution found: {0}'.format(
        np.array_str(result.params[0], max_line_width=np.inf,
                     precision=5, suppress_small=True)))


if __name__ == '__main__':
    main()
