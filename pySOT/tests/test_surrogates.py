from pySOT.surrogate import Surrogate, Tail, ConstantTail, LinearTail, \
    Kernel, CubicKernel, TPSKernel, LinearKernel, \
    GPRegressor, MARSInterpolant, PolyRegressor, RBFInterpolant, \
    SurrogateCapped, SurrogateUnitBox
from pySOT.optimization_problems import Ackley
import numpy.linalg as la
import numpy as np


def f(x):
    """Test function"""
    fx = x[:, 1] * np.sin(x[:, 0]) + x[:, 0] * np.cos(x[:, 1])
    fx = np.expand_dims(fx, axis=1)
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
    tail = LinearTail(dim=dim)
    assert(tail.degree == 1)
    assert(tail.dim_tail == dim + 1)
    X = np.random.rand(10, dim)
    np.testing.assert_allclose(tail.eval(X), np.hstack((np.ones((10, 1)), X)))
    x = X[0, :]
    np.testing.assert_allclose(tail.deriv(x), np.hstack((np.zeros((dim, 1)), np.eye(dim))))


def test_constant_tail():
    tail = ConstantTail(dim=1)
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
    rbf = RBFInterpolant(dim=2, eta=1e-6)
    assert (isinstance(rbf, Surrogate))
    fX = f(X)
    rbf.add_points(X, fX)

    # Derivative at random points
    np.random.seed(0)
    Xs = np.random.rand(10, 2)
    fhx = rbf.predict(Xs)
    dfhx = rbf.predict_deriv(Xs)
    fx = f(Xs)
    dfx = df(Xs)
    assert(np.max(np.abs(fx - fhx)) < 1e-4)
    assert(la.norm(dfx - dfhx) < 1e-2)

    # Derivative at previous points
    dfhx = rbf.predict_deriv(X[0, :])
    assert(la.norm(df(np.atleast_2d(X[0, :])) - dfhx) < 1e-1)

    # Reset the surrogate
    rbf.reset()
    assert(rbf.num_pts == 0 and rbf.dim == 2)

    # Now add 100 points at a time and test reallocation + LU
    for i in range(9):
        rbf.add_points(X[i*100:(i+1)*100, :], fX[i*100:(i+1)*100])
        rbf.predict(Xs)  # Force fit

    # Derivative at random points
    np.random.seed(0)
    Xs = np.random.rand(10, 2)
    fhx = rbf.predict(Xs)
    dfhx = rbf.predict_deriv(Xs)
    fx = f(Xs)
    dfx = df(Xs)
    assert(np.max(np.abs(fx - fhx)) < 1e-4)
    assert(la.norm(dfx - dfhx) < 1e-2)

    # Derivative at previous points
    dfhx = rbf.predict_deriv(X[0, :])
    assert(la.norm(df(np.atleast_2d(X[0, :])) - dfhx) < 1e-1)


def test_gp():
    X = make_grid(30)  # Make uniform grid with 30 x 30 points
    gp = GPRegressor(dim=2)
    assert (isinstance(gp, Surrogate))
    fX = f(X)
    gp.add_points(X, fX)

    # Derivative at random points
    np.random.seed(0)
    Xs = np.random.rand(10, 2)
    fhx = gp.predict(Xs)
    fx = f(Xs)
    assert(np.max(np.abs(fx - fhx)) < 1e-2)

    # Derivative at previous points
    # Reset the surrogate
    gp.reset()
    assert(gp.num_pts == 0 and gp.dim == 2)


def test_poly():
    X = make_grid(30)  # Make uniform grid with 30 x 30 points
    poly = PolyRegressor(dim=2, degree=2)
    assert(isinstance(poly, Surrogate))
    fX = f(X)
    poly.add_points(X, fX)

    # Derivative at random points
    np.random.seed(0)
    Xs = np.random.rand(10, 2)
    fhx = poly.predict(Xs)
    fx = f(Xs)
    assert(np.max(np.abs(fx - fhx)) < 1e-1)

    # Reset the surrogate
    poly.reset()
    assert (poly.num_pts == 0 and poly.dim == 2)


def test_mars():
    X = make_grid(30)  # Make uniform grid with 30 x 30 points
    try:
        mars = MARSInterpolant(dim=2)
    except Exception as e:
        print(str(e))
        return

    assert (isinstance(mars, Surrogate))
    fX = f(X)
    mars.add_points(X, fX)

    # Derivative at random points
    np.random.seed(0)
    Xs = np.random.rand(10, 2)
    fhx = mars.predict(Xs)
    fx = f(Xs)
    assert(np.max(np.abs(fx - fhx)) < 1e-1)

    # Reset the surrogate
    mars.reset()
    assert(mars.num_pts == 0 and mars.dim == 2)


def test_capped():
    def ff(x):
        return (6*x - 2)**2 * np.sin(12*x - 4)

    np.random.seed(0)
    x = np.random.rand(30, 1)
    fX = ff(x)

    xx = np.expand_dims(np.linspace(0, 1, 100), axis=1)

    # RBF with capping adapter
    rbf1 = SurrogateCapped(RBFInterpolant(dim=1, eta=1e-6))
    rbf1.add_points(x, fX)

    # RBF fitted to capped value
    fX_capped = fX.copy()
    fX_capped[fX > np.median(fX)] = np.median(fX)
    rbf2 = RBFInterpolant(dim=1, eta=1e-6)
    rbf2.add_points(x, fX_capped)

    assert(np.max(np.abs(rbf1.predict(xx) - rbf2.predict(xx))) < 1e-10)
    assert(np.max(np.abs(rbf1.predict_deriv(x[0, :]) - rbf2.predict_deriv(x[0, :]))) < 1e-10)

    rbf1.reset()
    assert(rbf1.num_pts == 0 and rbf1.dim == 1)
    assert(rbf1.X.size == 0 and rbf1.fX.size == 0)

def test_unit_box():
    ackley = Ackley(dim=1)
    np.random.seed(0)
    x = np.random.rand(30, 1)
    fX = np.expand_dims([ackley.eval(y) for y in x], axis=1)

    xx = np.expand_dims(np.linspace(0, 1, 100), axis=1)

    # RBF with internal scaling to unit hypercube
    rbf1 = SurrogateUnitBox(
        RBFInterpolant(dim=1, eta=1e-6), lb=np.array([0.0]), ub=np.array([1.0]))
    rbf1.add_points(x, fX)

    # Normal RBF
    rbf2 = RBFInterpolant(dim=1, eta=1e-6)
    rbf2.add_points(x, fX)

    assert(np.max(np.abs(rbf1.predict(xx) - rbf2.predict(xx))) < 1e-10)
    assert(np.max(np.abs(rbf1.predict_deriv(x[0, :]) - rbf2.predict_deriv(x[0, :]))) < 1e-10)
    assert(np.max(np.abs(rbf1.X - rbf2.X)) < 1e-10)
    assert(np.max(np.abs(rbf1.fX - rbf2.fX)) < 1e-10)

    rbf1.reset()
    assert(rbf1.num_pts == 0 and rbf1.dim == 1)
    assert(rbf1.X.size == 0 and rbf1.fX.size == 0)


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
    test_capped()
    test_unit_box()
