"""
.. module:: test_ensemble
  :synopsis: Test Ensemble surrogates
.. moduleauthor:: David Eriksson <dme65@cornell.edu>
"""

import logging
from pySOT import *
from poap.controller import ThreadController, BasicWorkerThread
import numpy as np
import os.path

def main():
    if not os.path.exists("./logfiles"):
        os.makedirs("logfiles")
    logging.basicConfig(filename="./logfiles/test_ensemble.log",
                        level=logging.INFO)

    print("\nNumber of threads: 4")
    print("Maximum number of evaluations: 50")
    print("Search strategy: CandidateSRBF")
    print("Experimental design: Latin Hypercube + point [0.1, 0.5, 0.8]")
    print("Surrogate: Cubic RBF, Linear RBF, Thin-plate RBF, MARS")

    nthreads = 4
    maxeval = 50
    nsamples = nthreads

    data = Hartman3()
    print(data.info)

    # Use 3 differents RBF's and MARS as an ensemble surrogate
    models = [
        RBFInterpolant(surftype=CubicRBFSurface, maxp=maxeval),
        RBFInterpolant(surftype=LinearRBFSurface, maxp=maxeval),
        RBFInterpolant(surftype=TPSSurface, maxp=maxeval)
    ]
    response_surface = EnsembleSurrogate(models, maxeval)

    # Add an additional point to the experimental design. If a good
    # solution is already known you can add this point to the
    # experimental design
    extra = np.atleast_2d([0.1, 0.5, 0.8])

    # Create a strategy and a controller
    controller = ThreadController()
    controller.strategy = \
        SyncStrategyNoConstraints(
            worker_id=0, data=data,
            response_surface=response_surface,
            maxeval=maxeval, nsamples=nsamples,
            exp_design=LatinHypercube(dim=data.dim, npts=2*(data.dim+1)),
            search_procedure=CandidateSRBF(data=data, numcand=100*data.dim),
            extra=extra)

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
    print('Best solution found: {0}\n'.format(
        np.array_str(result.params[0], max_line_width=np.inf,
                     precision=5, suppress_small=True)))


if __name__ == '__main__':
    main()
