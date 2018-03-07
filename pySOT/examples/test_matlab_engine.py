"""
.. module:: test_matlab_engine
  :synopsis: Test with MATLAB objective function
.. moduleauthor:: David Eriksson <dme65@cornell.edu>
"""

from pySOT.adaptive_sampling import CandidateDYCORS
from pySOT.experimental_design import LatinHypercube
from pySOT.strategy import SRBFStrategy
from pySOT.surrogate import RBFInterpolant, CubicKernel, LinearTail
from pySOT.optimization_problems import Ackley

from poap.controller import ThreadController, ProcessWorkerThread
import numpy as np
import os.path
import logging

# Try to import the matlab_wrapper module
try:
    import matlab_wrapper
except Exception as err:
    print("\nERROR: Failed to import the matlab_wrapper module. "
          "Install using: pip install matlab_wrapper\n")
    pass


class MatlabWorker(ProcessWorkerThread):
    def handle_eval(self, record):
        try:
            self.matlab.put('x', record.params[0])
            self.matlab.eval('matlab_ackley')
            val = self.matlab.get('val')
            if np.isnan(val):
                raise ValueError()
            self.finish_success(record, val)
        except:
            logging.info("WARNING: Incorrect output or crashed evaluation")
            self.finish_cancelled(record)


def test_matlab_engine():
    if not os.path.exists("./logfiles"):
        os.makedirs("logfiles")
    if os.path.exists("./logfiles/test_matlab_engine.log"):
        os.remove("./logfiles/test_matlab_engine.log")
    logging.basicConfig(filename="./logfiles/test_matlab_engine.log",
                        level=logging.INFO)

    print("\nNumber of threads: 4")
    print("Maximum number of evaluations: 500")
    print("Sampling method: CandidateDYCORS")
    print("Experimental design: Latin Hypercube")
    print("Surrogate: Cubic RBF")

    nthreads = 4
    maxeval = 500
    matlab_root = "/Applications/MATLAB_R2017b.app"

    opt_prob = Ackley(dim=10)
    print(opt_prob.info)

    surrogate = RBFInterpolant(dim=opt_prob.dim, kernel=CubicKernel(),
                               tail=LinearTail(opt_prob.dim), maxpts=maxeval)

    # Use the serial controller (uses only one thread)
    controller = ThreadController()
    controller.strategy = \
        SRBFStrategy(worker_id=0, maxeval=maxeval, opt_prob=opt_prob,
                     exp_design=LatinHypercube(dim=opt_prob.dim, npts=2*(opt_prob.dim+1)),
                     surrogate=surrogate,
                     sampling_method=CandidateDYCORS(data=opt_prob, numcand=100*opt_prob.dim),
                     batch_size=nthreads, async=False)

    print("\nNOTE: You may need to specify the matlab_root keyword in "
          "order \n      to start a MATLAB  session using the matlab_wrapper "
          "module\n")

    # We need to tell MATLAB where the script is
    mfile_location = os.getcwd()

    # Launch the threads
    for _ in range(nthreads):
        worker = MatlabWorker(controller)
        try:
            worker.matlab = matlab_wrapper.MatlabSession(options='-nojvm', matlab_root=matlab_root)
        except Exception as err:
            print("\nERROR: Failed to initialize a MATLAB session.\n")
            return

        worker.matlab.workspace.addpath(mfile_location)
        controller.launch_worker(worker)

    # Run the optimization strategy
    result = controller.run()

    # Print the final result
    print('Best value found: {0}'.format(result.value))
    print('Best solution found: {0}\n'.format(
        np.array_str(result.params[0], max_line_width=np.inf,
                     precision=5, suppress_small=True)))


if __name__ == '__main__':
    test_matlab_engine()
