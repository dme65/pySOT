Options
=======

Strategy
-----------------

We provide implementations of Stochastic RBF (SRBF), DYCORS, 
Expected Improvement (EI), lower confidence bound (LCB) and random search (RS).
EI can only be used in combination with GPRegressor since uncertainty predictions 
are necessary. All strategies support running in serial, batch synchronous parallel,
and asynchronous parallel. 

New optimization strategies can be implemented by inheriting from SurrogateBaseStrategy 
and implementing the abstract generate_evals method that proposes num_pts
new sample points:

- Required methods
    * generate_evals(num_pts): Proposes num_pts new samples.

Experimental design
-------------------

The experimental design generates the initial points to be evaluated. A well-chosen
experimental design is critical in order to fit a surrogate model that captures the behavior
of the underlying objective function. Any implementation must have the following attributes
and method:

- Attributes:
    * dim: Dimensionality
    * num_pts: Number of points in the design
- Required methods
    * generate_points(): Returns an experimental design of size num_pts x dim where
      num_pts is the number of points in the initial design, which was specified
      when the object was created.

The following experimental designs are supported:

- **LatinHypercube:**
    A Latin hypercube design

    Example:

    .. code-block:: python

        from pySOT import LatinHypercube
        exp_des = LatinHypercube(dim=3, npts=10)

    creates a Latin hypercube design with 10 points in 3 dimensions

- **SymmetricLatinHypercube**
    A symmetric Latin hypercube design

    Example:

    .. code-block:: python

        from pySOT import SymmetricLatinHypercube
        exp_des = SymmetricLatinHypercube(dim=3, npts=10)

    creates a symmetric Latin hypercube design with 10 points in 3 dimensions

- **TwoFactorial**
    The corners of the unit hypercube

    Example:

    .. code-block:: python

        from pySOT import TwoFactorial
        exp_des = TwoFactorial(dim=3)

    creates a symmetric Latin hypercube design with 8 points in 3 dimensions

Surrogate model
---------------

The surrogate model approximates the underlying objective function given all of the
points that have been evaluated. Any implementation of a surrogate model must
have the following attributes and methods

- Attributes:
    * dim: Number of dimensions
    * num_pts: Number of points in the surrogate model
    * X: Data points, of size num_pts x dim, currently incorporated in the model
    * fX: Function values at the data points
    * updated: True if all information is incorporated in the model, else a new fit will be triggered
- Required methods
    * reset(): Resets the surrogate model
    * add_points(x, fx): Adds point(s) x with value(s) fx to the surrogate model. This SHOULD NOT trigger a new fit of the model.
    * predict(x): Evaluates the surrogate model at points x
    * predict_deriv(x): Evaluates the derivative of surrogate model at points x
- Optional methods
    * predict_std(x): Evaluates the uncertainty of the surrogate model at points x

The following surrogate models are supported:

- **RBFInterpolant:**
    A radial basis function interpolant.

    Example:

    .. code-block:: python

        from pySOT.surrogate import RBFInterpolant, CubicKernel, LinearTail
        fhat = RBFInterpolant(dim=dim, kernel=CubicKernel(), tail=LinearTail(dim=dim))

    creates a cubic RBF with a linear tail in dim dimensions.

- **GPRegressor:**
    Generate a Gaussian process regression object.

    Example:

    .. code-block:: python

        from pySOT.surrogate import GPRegressor
        surrogate = GPRegressor(dim=dim)

    creates a GPRegressor object in dim dimensions.

- **MARSInterpolant:**
    Generate a Multivariate Adaptive Regression Splines (MARS) model.

    .. note:: This implementation depends on the py-earth module (see :ref:`quickstart-label`)

    Example:

    .. code-block:: python

        from pySOT.surrogate import MARSInterpolant
        surrogate = MARSInterpolant(dim=dim)

    creates a MARS interpolant in dim dimensions.

- **PolyRegressor:**
    Multivariate polynomial regression.

    Example:

    .. code-block:: python

        from pySOT.surrogate import PolyRegressor
        surrogate = PolyRegressor(dim=dim, degree=2)

    creates a polynomial regressor of degree 2.

Optimization problem
--------------------

The optimization problem is its own object and must have certain attributes and methods
in order to work with the framework. The following attributes and methods must
always be specified in the optimization problem class:

- **Attributes**
    * lb: Lower bounds for the variables.
    * ub: Upper bounds for the variables.
    * dim: Number of dimensions
    * int_var: Specifies the integer variables. If no variables have
      discrete, set to []
    * cont_var: Specifies the continuous variables. If no variables
      are continuous, set to []
- **Required methods**
    * eval: Takes one input in the form of an numpy.ndarray with
      shape (1, dim), which corresponds to one point in dim dimensions. Returns the
      value (a scalar) of the objective function at this point.
