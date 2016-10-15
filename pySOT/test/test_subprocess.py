"""
.. module:: test_subprocess
  :synopsis: Test an external objective function
.. moduleauthor:: David Eriksson <dme65@cornell.edu>
"""

import logging
from pySOT import *
from poap.controller import ThreadController, ProcessWorkerThread
import numpy as np
from subprocess32 import Popen, PIPE
import os.path


def array2str(x):
    return ",".join(np.char.mod('%f', x))


class CppSim(ProcessWorkerThread):
    def handle_eval(self, record):
        try:
            self.process = Popen(['./sphere_ext', array2str(record.params[0])], stdout=PIPE)
            val = self.process.communicate()[0]
            self.finish_success(record, float(val))
        except ValueError:
            self.finish_cancelled(record)
            logging.info("WARNING: Incorrect output or crashed evaluation")


def main():
    if not os.path.exists("./logfiles"):
        os.makedirs("logfiles")
    if os.path.exists("./logfiles/test_subprocess.log"):
        os.remove("./logfiles/test_subprocess.log")
    logging.basicConfig(filename="./logfiles/test_subprocess.log",
                        level=logging.INFO)

    print("\nNumber of threads: 4")
    print("Maximum number of evaluations: 200")
    print("Search strategy: Candidate DYCORS")
    print("Experimental design: Symmetric Latin Hypercube")
    print("Surrogate: Cubic RBF")

    assert os.path.isfile("./sphere_ext"), "You need to build sphere_ext"

    nthreads = 4
    maxeval = 200
    nsamples = nthreads

    data = Sphere(dim=10)
    print(data.info)

    # Create a strategy and a controller
    controller = ThreadController()
    controller.strategy = \
        SyncStrategyNoConstraints(
            worker_id=0, data=data,
            maxeval=maxeval, nsamples=nsamples,
            exp_design=SymmetricLatinHypercube(dim=data.dim, npts=2*(data.dim+1)),
            sampling_method=CandidateDYCORS(data=data, numcand=100*data.dim),
            response_surface=RBFInterpolant(kernel=CubicKernel, tail=LinearTail,
                                            maxp=maxeval, dim=data.dim))

    # Launch the threads and give them access to the objective function
    for _ in range(nthreads):
        controller.launch_worker(CppSim(controller))

    # Run the optimization strategy
    result = controller.run()

    print('Best value found: {0}'.format(result.value))
    print('Best solution found: {0}\n'.format(
        np.array_str(result.params[0], max_line_width=np.inf,
                     precision=5, suppress_small=True)))

if __name__ == '__main__':
    main()
