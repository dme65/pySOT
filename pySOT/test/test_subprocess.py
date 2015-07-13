"""
.. module:: test_subprocess
  :synopsis: Test an external objective function
.. moduleauthor:: David Eriksson <dme65@cornell.edu>
"""

import logging
from pySOT import *
from poap.controller import ThreadController, ProcessWorkerThread
import numpy as np
from subprocess import Popen, PIPE
import os.path

def array2str(x):
    return ",".join(np.char.mod('%f', x))


class SphereExt:
    def __init__(self, dim=10):
        self.xlow = -15 * np.ones(dim)
        self.xup = 20 * np.ones(dim)
        self.dim = dim
        self.info = str(dim)+"-dimensional Sphere function \n" +\
                             "Global optimum: f(0,0,...,0) = 0"
        self.min = 0
        self.integer = []
        self.continuous = np.arange(0, dim)


class DummySim(ProcessWorkerThread):

    def handle_eval(self, record):
        self.process = Popen(['./sphere_ext', array2str(record.params[0])],
                             stdout=PIPE)
        out = self.process.communicate()[0]
        try:
            val = float(out)  # This raises ValueError if out is not a float
            self.finish_success(record, val)
        except ValueError:
            logging.warning("Function evaluation crashed/failed")
            self.finish_failure(record)


def main():
    if not os.path.exists("./logfiles"):
        os.makedirs("logfiles")
    logging.basicConfig(filename="./logfiles/test_subprocess.log",
                        level=logging.INFO)

    print("\nNumber of threads: 4")
    print("Maximum number of evaluations: 200")
    print("Search strategy: Candidate DyCORS")
    print("Experimental design: Latin Hypercube")
    print("Ensemble surrogates: Cubic RBF")

    assert os.path.isfile("./sphere_ext"), "You need to build sphere_ext"

    nthreads = 4
    maxeval = 200
    nsamples = nthreads

    data = SphereExt(dim=10)
    print(data.info)

    # Create a strategy and a controller
    controller = ThreadController()
    controller.strategy = \
        SyncStrategyNoConstraints(
            worker_id=0, data=data,
            maxeval=maxeval, nsamples=nsamples,
            exp_design=LatinHypercube(dim=data.dim, npts=2*(data.dim+1)),
            search_procedure=CandidateDYCORS(data=data, numcand=100*data.dim),
            response_surface=RBFInterpolant(surftype=CubicRBFSurface, maxp=maxeval))

    # Launch the threads and give them access to the objective function
    for _ in range(nthreads):
        controller.launch_worker(DummySim(controller))

    # Run the optimization strategy
    result = controller.run()

    print('Best value found: {0}'.format(result.value))
    print('Best solution found: {0}\n'.format(
        np.array_str(result.params[0], max_line_width=np.inf,
                     precision=5, suppress_small=True)))

if __name__ == '__main__':
    main()
