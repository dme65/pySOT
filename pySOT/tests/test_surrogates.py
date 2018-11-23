from pySOT.surrogate import Surrogate, Tail, ConstantTail, LinearTail, \
    Kernel, CubicKernel, TPSKernel, LinearKernel, \
    GPRegressor, MARSInterpolant, PolyRegressor, RBFInterpolant

import inspect
import sys
import numpy.linalg as la
import numpy as np


def f(x):
    """Test function"""
    fx = x[:, 1] * np.sin(x[:, 0]) + x[:, 0] * np.cos(x[:, 1])
    return fx


def df(x):
    """Derivative of test function"""
    dfx = np.zeros((x.shape[0], 2))
    dfx[:, 0] = x[:, 1] * np.cos(x[:, 0]) + np.cos(x[:, 1])
    dfx[:, 1] = np.sin(x[:, 0]) - x[:, 0] * np.sin(x[:, 1])
    return dfx


def test_cubic_kernel():
    kernel = CubicKernel()
    assert(isinstance(kernel, Kernel))
    assert(kernel.order == 2)

    x = np.array(2)
    np.testing.assert_allclose(kernel.eval(x), 8)
    np.testing.assert_allclose(kernel.deriv(x), 12)

    X = np.random.rand(10, 3)
    np.testing.assert_allclose(kernel.eval(X), X ** 3)
    np.testing.assert_allclose(kernel.deriv(X), 3 * X ** 2)


def test_tps_kernel():
    kernel = TPSKernel()
    assert(isinstance(kernel, Kernel))
    assert(kernel.order == 2)

    x = np.array(2)
    np.testing.assert_allclose(kernel.eval(x), 4*np.log(2))
    np.testing.assert_allclose(kernel.deriv(x), 2*(1+2*np.log(2)))

    X = np.random.rand(10, 3)
    np.testing.assert_allclose(kernel.eval(X), X**2 * np.log(X))
    np.testing.assert_allclose(kernel.deriv(X), X * (1+2*np.log(X)))


def test_linear_kernel():
    kernel = LinearKernel()
    assert(isinstance(kernel, Kernel))
    assert(kernel.order == 1)

    x = np.array([2])
    np.testing.assert_allclose(kernel.eval(x), x)
    np.testing.assert_allclose(kernel.deriv(x), 1)

    X = np.random.rand(10, 3)
    np.testing.assert_allclose(kernel.eval(X), X)
    np.testing.assert_allclose(kernel.deriv(X), np.ones((10, 3)))


def test_linear_tail():
    tail = LinearTail(1)
    assert(isinstance(tail, Tail))

    x = np.array([2])
    np.testing.assert_allclose(tail.eval(x), np.array([[1, x]]))
    np.testing.assert_allclose(tail.deriv(x), np.array([[0, 1]]))

    dim = 3
    tail = LinearTail(dim)
    assert(tail.degree == 1)
    assert(tail.dim_tail == dim + 1)
    X = np.random.rand(10, dim)
    np.testing.assert_allclose(tail.eval(X), np.hstack((np.ones((10, 1)), X)))
    x = X[0, :]
    np.testing.assert_allclose(tail.deriv(x), np.hstack((np.zeros((dim, 1)), np.eye(dim))))


def test_constant_tail():
    tail = ConstantTail(1)
    assert(isinstance(tail, Tail))

    x = np.array([2])
    np.testing.assert_allclose(tail.eval(x), np.array([[1]]))
    np.testing.assert_allclose(tail.deriv(x), np.array([[0]]))

    dim = 3
    tail = ConstantTail(dim)
    assert(tail.degree == 0)
    assert(tail.dim_tail == 1)
    X = np.random.rand(10, dim)
    np.testing.assert_allclose(tail.eval(X), np.ones((10, 1)))
    x = X[0, :]
    np.testing.assert_allclose(tail.deriv(x), np.zeros((dim, 1)))


def make_grid(n):
    xv, yv = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    X = np.hstack((np.reshape(xv, (n*n, 1)), np.reshape(yv, (n*n, 1))))
    return X


def test_rbf():
    X = make_grid(30)  # Make uniform grid with 30 x 30 points
    rbf = RBFInterpolant(2, 500, eta=1e-6)
    assert (isinstance(rbf, Surrogate))
    fX = f(X)
    rbf.add_points(X, fX)

    # Derivative at random points
    np.random.seed(0)
    Xs = np.random.rand(10, 2)
    fhx = rbf.eval(Xs)
    dfhx = rbf.deriv(Xs)
    fx = f(Xs)
    dfx = df(Xs)
    for i in range(Xs.shape[0]):
        assert(abs(fx[i] - fhx[i]) < 1e-4)
        assert(la.norm(dfx[i, :] - dfhx[i]) < 1e-2)

    # Derivative at previous points
    dfhx = rbf.deriv(X[0, :])
    assert (la.norm(df(np.atleast_2d(X[0, :])) - dfhx) < 1e-1)

    # Reset the surrogate
    rbf.reset()
    rbf._maxpts = 500
    assert(rbf.npts == 0)
    assert(rbf.dim == 2)

    # Now add 100 points at a time and test reallocation + LU
    for i in range(9):
        rbf.add_points(X[i*100:(i+1)*100, :], fX[i*100:(i+1)*100])
        rbf.eval(Xs)  # Force fit

    # Derivative at random points
    np.random.seed(0)
    Xs = np.random.rand(10, 2)
    fhx = rbf.eval(Xs)
    dfhx = rbf.deriv(Xs)
    fx = f(Xs)
    dfx = df(Xs)
    for i in range(Xs.shape[0]):
        assert(abs(fx[i] - fhx[i]) < 1e-4)
        assert(la.norm(dfx[i, :] - dfhx[i]) < 1e-2)

    # Derivative at previous points
    dfhx = rbf.deriv(X[0, :])
    assert (la.norm(df(np.atleast_2d(X[0, :])) - dfhx) < 1e-1)


def test_gp():
    X = make_grid(30)  # Make uniform grid with 30 x 30 points
    gp = GPRegressor(2, 50)
    assert (isinstance(gp, Surrogate))
    fX = f(X)
    gp.add_points(X, fX)

    # Derivative at random points
    np.random.seed(0)
    Xs = np.random.rand(10, 2)
    fhx = gp.eval(Xs)
    fx = f(Xs)
    for i in range(Xs.shape[0]):
        assert (abs(fx[i] - fhx[i]) < 1e-2)

    # Derivative at previous points
    # Reset the surrogate
    gp.reset()
    gp._maxpts = 50
    assert (gp.npts == 0)
    assert (gp.dim == 2)


def test_poly():
    X = make_grid(30)  # Make uniform grid with 30 x 30 points
    poly = PolyRegressor(2, 500)
    assert (isinstance(poly, Surrogate))
    fX = f(X)
    poly.add_points(X, fX)

    # Derivative at random points
    np.random.seed(0)
    Xs = np.random.rand(10, 2)
    fhx = poly.eval(Xs)
    fx = f(Xs)
    for i in range(Xs.shape[0]):
        assert (abs(fx[i] - fhx[i]) < 1e-1)

    # Reset the surrogate
    poly.reset()
    poly._maxpts = 500
    assert (poly.npts == 0)
    assert (poly.dim == 2)


def test_mars():
    X = make_grid(30)  # Make uniform grid with 30 x 30 points
    try:
        mars = MARSInterpolant(2, 500)
    except Exception as e:
        print(str(e))
        return

    assert (isinstance(mars, Surrogate))
    fX = f(X)
    mars.add_points(X, fX)

    # Derivative at random points
    np.random.seed(0)
    Xs = np.random.rand(10, 2)
    fhx = mars.eval(Xs)
    fx = f(Xs)
    for i in range(Xs.shape[0]):
        assert (abs(fx[i] - fhx[i]) < 1e-1)

    # Reset the surrogate
    mars.reset()
    mars._maxpts = 500
    assert(mars.npts == 0)
    assert(mars.dim == 2)


if __name__ == '__main__':
    test_cubic_kernel()
    test_tps_kernel()
    test_linear_kernel()
    test_linear_tail()
    test_constant_tail()
    test_gp()
    test_mars()
    test_rbf()
    test_poly()
