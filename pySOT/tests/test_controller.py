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
import pytest

np.random.seed(0)
max_evals = 300
ackley = Ackley(dim=10)

fname = "checkpoint.pysot"


def check_strategy(controller):
    """Make sure the strategy object is correct."""

    # Check the strategy object
    assert controller.strategy.num_evals == controller.strategy.max_evals
    assert controller.strategy.pending_evals == 0
    assert controller.strategy.X.shape == \
        (controller.strategy.num_evals, ackley.dim)
    assert controller.strategy.fX.shape == (controller.strategy.num_evals, 1)
    assert controller.strategy.Xpend.shape == (0, ackley.dim)
    assert len(controller.strategy.fevals) == controller.strategy.num_evals

    # Check that the strategy and controller have the same information
    assert len(controller.fevals) == controller.strategy.num_evals
    for i in range(controller.strategy.num_evals):
        if controller.fevals[i].status == 'completed':
            idx = np.where((controller.strategy.X ==
                            controller.fevals[i].params[0]).all(axis=1))[0]

            assert(len(idx) == 1)
            assert np.all(controller.fevals[i].params[0] ==
                          controller.strategy.X[idx, :])
            assert controller.fevals[i].value == controller.strategy.fX[idx]
            assert np.all(controller.fevals[i].params[0] <= ackley.ub)
            assert np.all(controller.fevals[i].params[0] >= ackley.lb)


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
        surrogate=rbf, asynchronous=True)

    # Wrap controller in checkpoint object
    controller = CheckpointController(controller, fname=fname)
    controller.run()


def resume(controller):
    # Wrap controller in checkpoint object
    controller = CheckpointController(controller, fname=fname)
    result = controller.resume()
    assert(result.value < 2.0)  # To make sure performance is the same

    check_strategy(controller.controller)

    # Try to resume again and make sure an exception is raised
    with pytest.raises(IOError):
        result = controller.run()


if __name__ == '__main__':
    test_checkpoint_serial()
