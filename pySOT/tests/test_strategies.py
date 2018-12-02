from pySOT.strategy import SRBFStrategy, DYCORSStrategy, \
    ExpectedImprovementStrategy
from pySOT.experimental_design import SymmetricLatinHypercube
from pySOT.surrogate import GPRegressor, \
    RBFInterpolant, CubicKernel, LinearTail
from pySOT.optimization_problems import Ackley

from poap.controller import Monitor, SerialController, \
    ThreadController, BasicWorkerThread
import numpy as np
import pytest

num_threads = 4
ackley = Ackley(dim=10)


def check_strategy(controller):
    """Make sure the strategy object is correct."""

    # Check the strategy object
    assert controller.strategy.num_evals <= controller.strategy.max_evals
    assert controller.strategy.phase == 2
    assert controller.strategy.init_pending == 0
    assert controller.strategy.pending_evals == 0
    assert controller.strategy.X.shape == (controller.strategy.num_evals, ackley.dim)
    assert controller.strategy.fX.shape == (controller.strategy.num_evals, 1)
    assert controller.strategy.Xpend.shape == (0, ackley.dim)
    assert len(controller.strategy.fevals) == controller.strategy.num_evals

    # Check that the strategy and controller have the same information
    for i in range(controller.strategy.num_evals):
        idx = np.where((controller.strategy.X == \
            controller.fevals[i].params[0]).all(axis=1))[0]
        assert np.all(controller.fevals[i].params[0] == controller.strategy.X[idx, :])
        assert controller.fevals[i].value == controller.strategy.fX[idx]


def test_srbf_serial():
    max_evals = 200
    rbf = RBFInterpolant(
        dim=ackley.dim, kernel=CubicKernel(),
        tail=LinearTail(ackley.dim))
    slhd = SymmetricLatinHypercube(
        dim=ackley.dim, num_pts=2*(ackley.dim+1))

    # Create a strategy and a controller
    controller = SerialController(ackley.eval)
    controller.strategy = SRBFStrategy(
        max_evals=max_evals, opt_prob=ackley, exp_design=slhd,
        surrogate=rbf, asynchronous=True)
    controller.run()

    check_strategy(controller)


def test_srbf_sync():
    max_evals = 200
    rbf = RBFInterpolant(
        dim=ackley.dim, kernel=CubicKernel(),
        tail=LinearTail(ackley.dim))
    slhd = SymmetricLatinHypercube(
        dim=ackley.dim, num_pts=2*(ackley.dim+1))

    # Create a strategy and a controller
    controller = ThreadController()
    controller.strategy = SRBFStrategy(
        max_evals=max_evals, opt_prob=ackley, exp_design=slhd,
        surrogate=rbf, asynchronous=False, batch_size=num_threads)

    for _ in range(num_threads):
        worker = BasicWorkerThread(controller, ackley.eval)
        controller.launch_worker(worker)
    controller.run()

    check_strategy(controller)


def test_srbf_async():
    max_evals = 200
    rbf = RBFInterpolant(
        dim=ackley.dim, kernel=CubicKernel(),
        tail=LinearTail(ackley.dim))
    slhd = SymmetricLatinHypercube(
        dim=ackley.dim, num_pts=2*(ackley.dim+1))

    # Create a strategy and a controller
    controller = ThreadController()
    controller.strategy = SRBFStrategy(
        max_evals=max_evals, opt_prob=ackley, exp_design=slhd,
        surrogate=rbf, asynchronous=True, batch_size=None)

    for _ in range(num_threads):
        worker = BasicWorkerThread(controller, ackley.eval)
        controller.launch_worker(worker)
    controller.run()

    check_strategy(controller)


def test_dycors_serial():
    max_evals = 200
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
    controller.run()

    check_strategy(controller)


def test_dycors_sync():
    max_evals = 200
    rbf = RBFInterpolant(
        dim=ackley.dim, kernel=CubicKernel(),
        tail=LinearTail(ackley.dim))
    slhd = SymmetricLatinHypercube(
        dim=ackley.dim, num_pts=2*(ackley.dim+1))

    # Create a strategy and a controller
    controller = ThreadController()
    controller.strategy = DYCORSStrategy(
        max_evals=max_evals, opt_prob=ackley, exp_design=slhd,
        surrogate=rbf, asynchronous=False, batch_size=num_threads)

    for _ in range(num_threads):
        worker = BasicWorkerThread(controller, ackley.eval)
        controller.launch_worker(worker)
    controller.run()

    check_strategy(controller)


def test_dycors_async():
    max_evals = 200
    rbf = RBFInterpolant(
        dim=ackley.dim, kernel=CubicKernel(),
        tail=LinearTail(ackley.dim))
    slhd = SymmetricLatinHypercube(
        dim=ackley.dim, num_pts=2*(ackley.dim+1))

    # Create a strategy and a controller
    controller = ThreadController()
    controller.strategy = DYCORSStrategy(
        max_evals=max_evals, opt_prob=ackley, exp_design=slhd,
        surrogate=rbf, asynchronous=True, batch_size=None)

    for _ in range(num_threads):
        worker = BasicWorkerThread(controller, ackley.eval)
        controller.launch_worker(worker)
    controller.run()

    check_strategy(controller)



def test_ei_serial():
    max_evals = 50
    gp = GPRegressor(dim=ackley.dim)
    slhd = SymmetricLatinHypercube(
        dim=ackley.dim, num_pts=2*(ackley.dim+1))

    # Create a strategy and a controller
    controller = SerialController(ackley.eval)
    controller.strategy = ExpectedImprovementStrategy(
        max_evals=max_evals, opt_prob=ackley, exp_design=slhd,
        surrogate=gp, asynchronous=True)
    controller.run()

    check_strategy(controller)


def test_ei_sync():
    max_evals = 50
    gp = GPRegressor(dim=ackley.dim)
    slhd = SymmetricLatinHypercube(
        dim=ackley.dim, num_pts=2*(ackley.dim+1))

    # Create a strategy and a controller
    controller = ThreadController()
    controller.strategy = DYCORSStrategy(
        max_evals=max_evals, opt_prob=ackley, exp_design=slhd,
        surrogate=gp, asynchronous=False, batch_size=num_threads)

    for _ in range(num_threads):
        worker = BasicWorkerThread(controller, ackley.eval)
        controller.launch_worker(worker)
    controller.run()

    check_strategy(controller)


def test_ei_async():
    max_evals = 50
    gp = GPRegressor(dim=ackley.dim)
    slhd = SymmetricLatinHypercube(
        dim=ackley.dim, num_pts=2*(ackley.dim+1))

    # Create a strategy and a controller
    controller = ThreadController()
    controller.strategy = DYCORSStrategy(
        max_evals=max_evals, opt_prob=ackley, exp_design=slhd,
        surrogate=gp, asynchronous=True, batch_size=None)

    for _ in range(num_threads):
        worker = BasicWorkerThread(controller, ackley.eval)
        controller.launch_worker(worker)
    controller.run()

    check_strategy(controller)


if __name__ == '__main__':
    test_srbf_serial()
    test_srbf_sync()
    test_srbf_async()
    test_dycors_serial()
    test_dycors_sync()
    test_dycors_async()
    test_ei_serial()
    test_ei_sync()
    test_ei_async()