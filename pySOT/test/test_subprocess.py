"""
.. module:: test_subprocess
  :synopsis: Test an external objective function
.. moduleauthor:: David Eriksson <dme65@cornell.edu>
"""

from pySOT import *
from poap.controller import ThreadController, ProcessWorkerThread
from poap.strategy import MultiStartStrategy
import numpy as np
from subprocess import Popen, PIPE

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
            val = float(out)
            self.finish_success(record, val)
        except:
            self.finish_failure(record)

def main():
    print("Number of threads: 4")
    print("Maximum number of evaluations: 200")
    print("Search strategy: Candidate DyCORS")
    print("Experimental design: Latin Hypercube")
    print("Ensemble surrogates: Cubic RBF")

    nthreads = 4
    maxeval = 200
    nsamples = nthreads

    data = SphereExt(dim=3)
    print(data.info)

    exp_design = LatinHypercube(dim=data.dim, npts=2*data.dim+1)
    response_surface = RBFInterpolant(phi=phi_cubic, P=linear_tail,
                                      dphi=dphi_cubic, dP=dlinear_tail,
                                      eta=1e-8, maxp=maxeval)
    search_proc = CandidateDyCORS(data=data, numcand=200*data.dim)

    # Create a strategy and a controller
    controller = ThreadController()
    strategy = [SyncStrategyNoConstraints(worker_id=0, data=data,
                                          response_surface=response_surface,
                                          maxeval=maxeval, nsamples=nsamples,
                                          exp_design=exp_design,
                                          search_procedure=search_proc,
                                          quiet=False)]
    strategy = MultiStartStrategy(controller, strategy)
    controller.strategy = strategy

    # Launch the threads and give them access to the objective function
    for _ in range(nthreads):
        controller.launch_worker(DummySim(controller))

    # Run the optimization strategy
    result = controller.run()

    print('Best value found: ' + str(result.value))
    print('Best solution found: ' + np.array_str(result.params[0], max_line_width=np.inf,
                                                 precision=5, suppress_small=True))

if __name__ == '__main__':
    main()
