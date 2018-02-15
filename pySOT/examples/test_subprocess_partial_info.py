"""
.. module:: test_subprocess_partial_info
  :synopsis: Test an external objective function with partial info
.. moduleauthor:: David Eriksson <dme65@cornell.edu>
"""

from pySOT.adaptive_sampling import CandidateDYCORS
from pySOT.experimental_design import SymmetricLatinHypercube
from pySOT.strategy import SyncStrategyNoConstraints
from pySOT.surrogate import RBFInterpolant, CubicKernel, LinearTail

from poap.controller import ThreadController, ProcessWorkerThread
import numpy as np
import sys
import os.path
import logging
from pySOT.optimization_problems import OptimizationProblem

if sys.version_info < (3, 0):
    # Try to import from subprocess32
    try:
        from subprocess32 import Popen, PIPE
    except Exception as err:
        print("ERROR: You need the subprocess32 module for Python 2.7. \n"
              "Install using: pip install subprocess32")
        exit()
else:
    from subprocess import Popen, PIPE


def array2str(x):
    return ",".join(np.char.mod('%f', x))


# Find path of the executable
path = os.path.dirname(os.path.abspath(__file__)) + "/sumfun_ext"


class SumfunExt(OptimizationProblem):
    def __init__(self, dim=10):
        self._dim = dim
        self.info = str(dim)+"-dimensional Sumfun function \n" +\
                             "Global optimum: f(0,0,...,0) = 0"
        self.min = 0

    @property
    def dim(self):
        return self._dim

    @property
    def lb(self):
        return -5 * np.ones(self.dim)

    @property
    def ub(self):
        return 5 * np.ones(self.dim)

    @property
    def nexp(self):
        return 0

    @property
    def ncheap(self):
        return 0

    @property
    def int_var(self):
        return np.array([])

    @property
    def cont_var(self):
        return np.arange(0, self.dim)

    def eval_cheap(self, X):
        raise NotImplementedError("There are no cheap constraints")

    def deval_cheap(self, X):
        raise NotImplementedError("There are no cheap constraints")

    def eval(self, xx):
        pass


class CppSim(ProcessWorkerThread):

    def handle_eval(self, record):
        val = np.nan
        # Continuously check for new outputs from the subprocess
        self.process = Popen([path, array2str(record.params[0])], stdout=PIPE, bufsize=1, universal_newlines=True)

        for line in self.process.stdout:
            try:
                val = float(line.strip())  # Try to parse output
                if val > 350:  # Terminate if too large
                    self.process.terminate()
                    self.finish_success(record, 350)
                    return
            except ValueError:  # If the output is nonsense we terminate
                logging.warning("Incorrect output")
                self.process.terminate()
                self.finish_cancelled(record)
                return
        self.process.wait()

        rc = self.process.poll()  # Check the return code
        if rc < 0 or np.isnan(val):
            logging.info("WARNING: Incorrect output or crashed evaluation")
            self.finish_cancelled(record)
        else:
            self.finish_success(record, val)


def test_subprocess_partial_info():
    if not os.path.exists("./logfiles"):
        os.makedirs("logfiles")
    if os.path.exists("./logfiles/test_subprocess_partial_info.log"):
        os.remove("./logfiles/test_subprocess_partial_info.log")
    logging.basicConfig(filename="./logfiles/test_subprocess_partial_info.log",
                        level=logging.INFO)

    print("\nNumber of threads: 4")
    print("Maximum number of evaluations: 200")
    print("Sampling method: Candidate DYCORS")
    print("Experimental design: Symmetric Latin Hypercube")
    print("Surrogate: Cubic RBF")

    assert os.path.isfile(path), "You need to build sumfun_ext"

    nthreads = 4
    maxeval = 200
    nsamples = nthreads

    data = SumfunExt(dim=10)
    print(data.info)

    # Create a strategy and a controller
    controller = ThreadController()
    controller.strategy = \
        SyncStrategyNoConstraints(
            worker_id=0, data=data,
            maxeval=maxeval, nsamples=nsamples,
            exp_design=SymmetricLatinHypercube(dim=data.dim, npts=2*(data.dim+1)),
            sampling_method=CandidateDYCORS(data=data, numcand=100*data.dim),
            response_surface=RBFInterpolant(dim=data.dim, kernel=CubicKernel(),
                                            tail=LinearTail(data.dim), maxpts=maxeval))

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
    test_subprocess_partial_info()
