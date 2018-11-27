from pySOT.experimental_design import SymmetricLatinHypercube
from pySOT.strategy import DYCORSStrategy
from pySOT.surrogate import RBFInterpolant, CubicKernel, LinearTail
from pySOT.optimization_problems import Ackley
from pySOT.controller import CheckpointController

from poap.controller import SerialController
import numpy as np
import multiprocessing
import time
import os

np.random.seed(0)
max_evals = 300
ackley = Ackley(dim=10)

fname = "checkpoint.pysot"

def test_checkpoint_serial():
    if os.path.isfile(fname):
        os.remove(fname)

    # Run for 1 seconds and kill the controller
    p = multiprocessing.Process(target=init_serial, args=())
    p.start()
    time.sleep(3)
    p.terminate()
    p.join()

    # Resume the run
    controller = SerialController(ackley.eval)
    resume(controller)


def init_serial():
    rbf = RBFInterpolant(
        dim=ackley.dim, kernel=CubicKernel(),
        tail=LinearTail(ackley.dim))
    slhd = SymmetricLatinHypercube(
        dim=ackley.dim, num_pts=2*(ackley.dim+1))

    # Create a strategy and a controller
    controller = SerialController(ackley.eval)
    controller.strategy = DYCORSStrategy(
        max_evals=max_evals, opt_prob=ackley, exp_design=slhd,
        surrogate=rbf, asynchronous=True, extra=None)

    # Wrap controller in checkpoint object
    controller = CheckpointController(controller, fname=fname)
    controller.run()


def resume(controller):
    # Wrap controller in checkpoint object
    controller = CheckpointController(controller, fname=fname)
    result = controller.resume()
    assert(result.value < 2.0)

if __name__ == '__main__':
    test_checkpoint_serial()