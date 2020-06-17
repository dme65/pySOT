import numpy as np

from pySOT.auxiliary_problems import candidate_srbf, candidate_uniform, ei_ga
from pySOT.optimization_problems import Ackley
from pySOT.surrogate import GPRegressor


def test_srbf():
    np.random.seed(0)
    ackley = Ackley(dim=1)
    X = np.expand_dims([-15, -10, 0, 1, 20], axis=1)
    fX = np.array([ackley.eval(x) for x in X])

    gp = GPRegressor(dim=ackley.dim, lb=ackley.lb, ub=ackley.ub)
    gp.add_points(X, fX)

    # Find the next point with w = 0.25
    x_true = 10.50
    x_next = candidate_uniform(
        num_pts=1, X=X, Xpend=None, fX=fX, num_cand=10000, surrogate=gp, opt_prob=ackley, weights=[0.25]
    )
    assert np.isclose(x_next, x_true, atol=1e-2)

    x_next = candidate_srbf(
        num_pts=1,
        X=X,
        Xpend=None,
        fX=fX,
        num_cand=10000,
        surrogate=gp,
        opt_prob=ackley,
        weights=[0.25],
        sampling_radius=0.5,
    )
    assert np.isclose(x_next, x_true, atol=1e-2)

    # Find the next point with w = 0.75
    x_true = -1.5050
    x_next = candidate_uniform(
        num_pts=1, X=X, Xpend=None, fX=fX, num_cand=10000, surrogate=gp, opt_prob=ackley, weights=[0.75]
    )
    assert np.isclose(x_next, x_true, atol=1e-2)

    x_next = candidate_srbf(
        num_pts=1,
        X=X,
        Xpend=None,
        fX=fX,
        num_cand=10000,
        surrogate=gp,
        opt_prob=ackley,
        weights=[0.75],
        sampling_radius=0.5,
    )
    assert np.isclose(x_next, x_true, atol=1e-2)


def test_ei():
    np.random.seed(0)
    ackley = Ackley(dim=1)
    X = np.expand_dims([-15, -10, 0, 1, 20], axis=1)
    fX = np.array([ackley.eval(x) for x in X])

    gp = GPRegressor(dim=ackley.dim, lb=ackley.lb, ub=ackley.ub)
    gp.add_points(X, fX)

    # Find the global optimizer of EI
    x_true = -1.7558
    x_next = ei_ga(X=X, Xpend=None, dtol=0.0, ei_tol=0, fX=fX, num_pts=1, opt_prob=ackley, surrogate=gp)
    assert np.isclose(x_next, x_true, atol=1e-2)

    # Find the optimizer at least distance 5 from other points
    x_true = 10.6656
    x_next = ei_ga(X=X, Xpend=None, dtol=5.0, ei_tol=0, fX=fX, num_pts=1, opt_prob=ackley, surrogate=gp)
    assert np.isclose(x_next, x_true, atol=1e-2)


if __name__ == "__main__":
    test_ei()
    test_srbf()
