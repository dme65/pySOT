"""
.. module:: test_matlab_engine
  :synopsis: Test with MATLAB objective function
.. moduleauthor:: David Eriksson <dme65@cornell.edu>
"""

from pySOT import *
from poap.controller import SerialController
import numpy as np
import os.path
import matlab_wrapper


matlab = matlab_wrapper.MatlabSession(options='-nojvm')
#matlab = matlab_wrapper.MatlabSession(matlab_root='/Applications/MATLAB_R2014a.app', options='-nojvm')

# You can try to specify the location of the matlabroot if the session doesn't start
# On OSX it's necessary to specify this. Typing matlabroot in a MATLAB session tells you where the root
# folder is. The following works on OSX:
#   matlab = matlab_wrapper.MatlabSession(matlab_root='/Applications/MATLAB_R2014a.app', options='-nojvm')


class AckleyExt:
    def __init__(self, dim=10):
        self.xlow = -15 * np.ones(dim)
        self.xup = 20 * np.ones(dim)
        self.dim = dim
        self.info = str(dim) + "-dimensional Ackley function \n" + \
            "Global optimum: f(0,0,...,0) = 0"
        self.min = 0
        self.integer = []
        self.continuous = np.arange(0, dim)
        check_opt_prob(self)

    def objfunction(self, x):
        matlab.put('x', x)
        matlab.eval('matlab_ackley')
        val = matlab.get('val')
        return val


def main():
    if not os.path.exists("./logfiles"):
        os.makedirs("logfiles")
    if os.path.exists("./logfiles/test_matlab_engine.log"):
        os.remove("./logfiles/test_matlab_engine.log")
    logging.basicConfig(filename="./logfiles/test_matlab_engine.log",
                        level=logging.INFO)

    print("\nNumber of threads: 1")
    print("Maximum number of evaluations: 500")
    print("Search strategy: CandidateDYCORS")
    print("Experimental design: Latin Hypercube")
    print("Surrogates: Cubic RBF, domain scaled to unit box")

    maxeval = 500

    data = AckleyExt(dim=10)
    print(data.info)

    # Use the serial controller (uses only one thread)
    controller = SerialController(data.objfunction)
    controller.strategy = \
        SyncStrategyNoConstraints(
            worker_id=0, data=data,
            maxeval=maxeval, nsamples=1,
            exp_design=LatinHypercube(dim=data.dim, npts=2*(data.dim+1)),
            response_surface=RSUnitbox(RBFInterpolant(surftype=CubicRBFSurface, maxp=maxeval),data),
            sampling_method=CandidateDYCORS(data=data, numcand=100*data.dim))

    # Run the optimization strategy
    result = controller.run()

    # Print the final result
    print('Best value found: {0}'.format(result.value))
    print('Best solution found: {0}'.format(
        np.array_str(result.params[0], max_line_width=np.inf,
                     precision=5, suppress_small=True)))


if __name__ == '__main__':
    main()
