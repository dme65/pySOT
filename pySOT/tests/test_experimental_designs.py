from pySOT.experimental_design import ExperimentalDesign, \
        SymmetricLatinHypercube, LatinHypercube, TwoFactorial
import numpy as np
import pytest


def test_lhd():
    lhd = LatinHypercube(dim=4, num_pts=10)
    X = lhd.generate_points()
    assert isinstance(lhd, ExperimentalDesign)
    assert np.all(X.shape == (10, 4))
    assert lhd.num_pts == 10
    assert lhd.dim == 4


def test_lhd_round():
    num_pts = 10
    dim = 3
    lb = np.array([1, 2, 3])
    ub = np.array([3, 4, 5])
    int_var = np.array([1])

    np.random.seed(0)
    lhd = LatinHypercube(dim=dim, num_pts=num_pts)
    X = lhd.generate_points(lb=lb, ub=ub, int_var=int_var)

    assert np.all(np.round(X[:, 1] == X[:, 1]))  # Should be integers
    assert np.all(np.max(X, axis=0) <= ub)
    assert np.all(np.min(X, axis=0) >= lb)


def test_slhd():
    for i in range(10, 12):  # To test even and odd
        slhd = SymmetricLatinHypercube(dim=3, num_pts=i)
        X = slhd.generate_points()
        assert isinstance(slhd, ExperimentalDesign)
        assert np.all(X.shape == (i, 3))
        assert slhd.num_pts == i
        assert slhd.dim == 3


def test_slhd_round():
    num_pts = 10
    dim = 3
    lb = np.array([1, 2, 3])
    ub = np.array([3, 4, 5])
    int_var = np.array([1])

    np.random.seed(0)
    slhd = SymmetricLatinHypercube(dim=dim, num_pts=num_pts)
    X = slhd.generate_points(lb=lb, ub=ub, int_var=int_var)
    assert np.all(np.round(X[:, 1] == X[:, 1]))  # Should be integers
    assert np.all(np.max(X, axis=0) == ub)
    assert np.all(np.min(X, axis=0) == lb)


def test_full_factorial():
    ff = TwoFactorial(dim=3)
    X = ff.generate_points()
    assert isinstance(ff, ExperimentalDesign)
    assert np.all(X.shape == (8, 3))
    assert ff.num_pts == 8
    assert ff.dim == 3
    assert np.all(np.logical_or(X == 1, X == 0))

    with pytest.raises(ValueError):  # This should raise an exception
        TwoFactorial(20)


def test_full_factorial_round():
    lb = np.array([1, 2, 3])
    ub = np.array([3, 4, 5])
    int_var = np.array([1])

    ff = TwoFactorial(dim=3)
    X = ff.generate_points(lb=lb, ub=ub, int_var=int_var)
    assert np.all(np.logical_or(X == lb, X == ub))


if __name__ == '__main__':
    test_full_factorial()
    test_lhd()
    test_slhd()

    test_lhd_round()
    test_slhd_round()
    test_full_factorial_round()
