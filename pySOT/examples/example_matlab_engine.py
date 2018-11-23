"""
.. module:: example_matlab_engine
  :synopsis: Example with MATLAB objective function
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


def example_matlab_engine():
    if not os.path.exists("./logfiles"):
        os.makedirs("logfiles")
    if os.path.exists("./logfiles/example_matlab_engine.log"):
        os.remove("./logfiles/example_matlab_engine.log")
    logging.basicConfig(filename="./logfiles/example_matlab_engine.log",
                        level=logging.INFO)

    print("\nNumber of threads: 4")
    print("Maximum number of evaluations: 500")
    print("Sampling method: CandidateDYCORS")
    print("Experimental design: Latin Hypercube")
    print("Surrogate: Cubic RBF")

    nthreads = 4
    max_evals = 500
    matlab_root = "/Applications/MATLAB_R2018b.app"

    ackley = Ackley(dim=10)
    print(ackley.info)

    rbf = RBFInterpolant(dim=ackley.dim, kernel=CubicKernel(),
                         tail=LinearTail(ackley.dim), maxpts=max_evals)
    dycors = CandidateDYCORS(opt_prob=ackley, max_evals=max_evals, numcand=100*ackley.dim)
    slhd = LatinHypercube(dim=ackley.dim, npts=ackley.dim+1)

    # Use the serial controller (uses only one thread)
    controller = ThreadController()
    controller.strategy = \
        SRBFStrategy(max_evals=max_evals, opt_prob=ackley, asynchronous=False,
                    exp_design=slhd, surrogate=rbf, adapt_sampling=dycors,
                    batch_size=nthreads)

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
        except:
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
    example_matlab_engine()
