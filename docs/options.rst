Options
=======

Optimization problem
--------------------

Experimental design
-------------------

Surrogate model
---------------

The surrogate model approximates the underlying objective function given all of the
points that have been evaluated. Any implementation of a surrogate model must
have the following attributes and methods

- Attributes:
    * nump: Number of data points (integer)
    * maxp: Maximum number of data points (integer)
- Required methods
    * reset(): Resets the surrogate model
    * get_x(): Returns a numpy array of size nump x d of the data points
    * get_fx(): Returns a numpy array of length nump with the function values
    * add_point(x, f): Adds a point x with value f to the surrogate model
    * eval(x): Evaluates the surrogate model at one point x
    * evals(x): Evaluates the surrogate model at multiple points
- Optional methods
    * deriv(x): Returns a numpy array with the gradient at one point x

The following surrogate models are supported:

- **RBFInterpolant:**
    A radial basis function interpolant.

    Example:

    .. code-block:: python

        from pySOT import RBFInterpolant, CubicRBFSurface
        fhat = RBFInterpolant(kernel=CubicKernel, tail=LinearTail, maxp=500)
    creates a cubic RBF with a linear tail with a capacity for 500 points.

- **KrigingInterpolant:**
    Generate a Kriging interpolant.

    Example:

    .. code-block:: python

        from pySOT import KrigingInterpolant
        fhat = KrigingInterpolant(maxp=500)
    creates a KrigingInterpolant interpolant with a capacity of 500 points.

- **MARSInterpolant:**
    Generate a Multivariate Adaptive Regression Splines (MARS) model.

    Example:

    .. code-block:: python

        from pySOT import MARSInterpolant
        fhat = MARSInterpolant(maxp=500)
    creates a MARS interpolant with a capacity of 500 points.

- **PolyRegression:**
    Multivariate polynomial regression.

    Example:

    .. code-block:: python

        from pySOT import PolyRegression
        bounds = bounds = np.hstack((np.zeros((3,1)), np.ones((3,1)))) # Our points are in [0,1]^3
        basisp = basis_TD(3, 2)  # use degree 2 with cross-terms
        fhat = PolyRegression(bounds=bounds, basisp=basisp, maxp=500)
    creates a polynomial regression surface of degree 2 with no cross-terms
    interpolant and a capacity of 500 points.

- **EnsembleSurrogate:**
    We also provide the option of using multiple surrogates
    for the same problem. Suppose we have M surrogate models, then the ensemble
    surrogate takes the form

    .. math::
        s(x) = \sum_{j=1}^M w_j s_j(x)

    where :math:`w_j` are non-negative weights that sum to 1. Hence the value of the ensemble
    surrogate is the weighted prediction of the M surrogate models. We use leave-one-out
    for each surrogate model to predict the function value at the removed point and then
    compute several statistics such as correlation with the true function values, RMSE, etc.
    Based on these statistics we use Dempster-Shafer Theory to compute the pignistic
    probability for each model, and take this probability as the weight. Surrogate models
    that does a good job predicting the removed points will generally be given a large
    weight.

    Example:

    .. code-block:: python

        from pySOT import RBFInterpolant, CubicRBFSurface, LinearRBFSurface,
            TPSSurface, EnsembleSurrogate

        models = [
            RBFInterpolant(surftype=CubicRBFSurface, maxp=500),
            RBFInterpolant(surftype=LinearRBFSurface, maxp=500),
            RBFInterpolant(surftype=TPSSurface, maxp=500)
        ]

        response_surface = EnsembleSurrogate(model_list=models, maxp=500)
    creates an ensemble surrogate with three surrogate models, namely a
    Cubic RBF Interpolant, a Linear RBF Interpolant, and a TPS RBF Interpolant.

Adaptive sampling
-----------------

