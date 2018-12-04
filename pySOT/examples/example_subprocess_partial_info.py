"""
.. module:: example_subprocess_partial_info
  :synopsis: Example an external objective function with partial info
.. moduleauthor:: David Eriksson <dme65@cornell.edu>
"""

from pySOT.experimental_design import SymmetricLatinHypercube
from pySOT.strategy import SRBFStrategy
from pySOT.surrogate import RBFInterpolant, CubicKernel, LinearTail

from poap.controller import ThreadController, ProcessWorkerThread
import numpy as np
import sys
import os.path
import logging
from pySOT.optimization_problems import OptimizationProblem
from subprocess import Popen, PIPE


def array2str(x):
    return ",".join(np.char.mod('%f', x))

# Find path of the executable
path = os.path.dirname(os.path.abspath(__file__)) + "/sumfun_ext"


class SumfunExt(OptimizationProblem):
    def __init__(self, dim=10):
        self.dim = dim
        self.lb = -5 * np.ones(self.dim)
        self.ub = 5 * np.ones(self.dim)
        self.cont_var = np.arange(0, self.dim)
        self.int_var = np.array([])
        self.info = str(dim) + "-dimensional Sumfun function \n" +\
                               "Global optimum: f(0,0,...,0) = 0"
        self.min = 0

    def eval(self, xx):
        pass


class CppSim(ProcessWorkerThread):

    def handle_eval(self, record):
        val = np.nan
        # Continuously check for new outputs from the subprocess
        self.process = Popen([path, array2str(record.params[0])],
                             stdout=PIPE, bufsize=1, universal_newlines=True)

        for line in self.process.stdout:
            try:
                val = float(line.strip())  # Try to parse output
                if val > 250.0:  # Terminate if too large
                    self.process.terminate()
                    self.finish_success(record, 250.0)
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


def example_subprocess_partial_info():
    if not os.path.exists("./logfiles"):
        os.makedirs("logfiles")
    if os.path.exists("./logfiles/example_subprocess_partial_info.log"):
        os.remove("./logfiles/example_subprocess_partial_info.log")
    logging.basicConfig(
        filename="./logfiles/example_subprocess_partial_info.log",
        level=logging.INFO)

    assert os.path.isfile(path), "You need to build sumfun_ext"

    num_threads = 4
    max_evals = 200

    sumfun = SumfunExt(dim=10)
    rbf = RBFInterpolant(
        dim=sumfun.dim, kernel=CubicKernel(),
        tail=LinearTail(sumfun.dim))
    slhd = SymmetricLatinHypercube(
        dim=sumfun.dim, num_pts=2*(sumfun.dim+1))

    # Create a strategy and a controller
    controller = ThreadController()
    controller.strategy = SRBFStrategy(
        max_evals=max_evals, opt_prob=sumfun, exp_design=slhd,
        surrogate=rbf, asynchronous=True, batch_size=num_threads)

    print("Number of threads: {}".format(num_threads))
    print("Maximum number of evaluations: {}".format(max_evals))
    print("Strategy: {}".format(controller.strategy.__class__.__name__))
    print("Experimental design: {}".format(slhd.__class__.__name__))
    print("Surrogate: {}".format(rbf.__class__.__name__))

    # Launch the threads and give them access to the objective function
    for _ in range(num_threads):
        controller.launch_worker(CppSim(controller))

    # Run the optimization strategy
    result = controller.run()

    print('Best value found: {0}'.format(result.value))
    print('Best solution found: {0}\n'.format(
        np.array_str(result.params[0], max_line_width=np.inf,
                     precision=5, suppress_small=True)))


if __name__ == '__main__':
    example_subprocess_partial_info()
