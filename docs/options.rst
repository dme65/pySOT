Options
=======

Optimization problem
--------------------

The optimization problem is its own object and must have certain attributes and methods
in order to work with the framework. We start by giving an example of a mixed-integer
optimization problem with constraints. The following attributes and methods must
always be specified in the optimization problem class:

- **Attributes**
    * xlow: Lower bounds for the variables.
    * xup: Upper bounds for the variables.
    * dim: Number of dimensions
    * integer: Specifies the integer variables. If no variables have
      discrete, set to []
    * continuous: Specifies the continuous variables. If no variables
      are continuous, set to []
- **Required methods**
    * objfunction: Takes one input in the form of an numpy.ndarray with
      shape (1, dim), which corresponds to one point in dim dimensions. Returns the
      value (a scalar) of the objective function at this point.
- **Optional methods**
    * eval_ineq_constraints:  Only necessary if there are inequality constraints.
      All constraints must be inequality constraints and the must be written in the form
      :math:`g_i(x) \leq 0`. The function takes one input in the form of an numpy.ndarray of
      shape (n, dim), which corresponds to n points in dim dimensions. Returns an
      numpy.ndarray of shape (n, M) where M is the number of inequality constraints.
    * deriv_ineq_constraints: Only necessary if there are inequality constraints and
      an adaptive sampling method that requires gradient information of the constraints is
      used. Returns a numpy ndarray of shape (n, nconstraints, dim)

What follows is an example of an objective function in 5 dimensions with 3 integer and 2
continuous variables. There are also 3 inequality constraints that are not bound constraints
which means that we need to implement the eval_ineq_constraints method.

.. code-block:: python

    import numpy as np

    class LinearMI:
        def __init__(self):
            self.xlow = np.zeros(5)
            self.xup = np.array([10, 10, 10, 1, 1])
            self.dim = 5
            self.min = -1
            self.integer = np.arange(0, 3)
            self.continuous = np.arange(3, 5)

        def eval_ineq_constraints(self, x):
            vec = np.zeros((x.shape[0], 3))
            vec[:, 0] = x[:, 0] + x[:, 2] - 1.6
            vec[:, 1] = 1.333 * x[:, 1] + x[:, 3] - 3
            vec[:, 2] = - x[:, 2] - x[:, 3] + x[:, 4]
            return vec

        def objfunction(self, x):
            if len(x) != self.dim:
                raise ValueError('Dimension mismatch')
            return - x[0] + 3 * x[1] + 1.5 * x[2] + 2 * x[3] - 0.5 * x[4]

**Note:** The method check_opt_prob which is available in pySOT is helpful in order t
o test that the objective function is compatible with the framework.

Experimental design
-------------------

The experimental design generates the initial points to be evaluated. A well-chosen
experimental design is critical in order to fit a surrogate model that captures the behavior
of the underlying objective function. Any implementation must have the following attributes
and method:

- Attributes:
    * dim: Dimensionality
    * npts: Number of points in the design
- Required methods
    * generate_points(): Returns an experimental design of size npts x d where
      npts is the number of points in the initial design, which was specified
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

- **BoxBehnken**.
    Box-Behnken design with one center point. This means that the design consits
    of the midpoints of the edges of the unit hypercube plus the center of the unit
    hypercube.

    Example:

    .. code-block:: python

        from pySOT import BoxBehnken
        exp_des = BoxBehnken(dim=3)

    creates a Box-Behnken design with 13 points in 3 dimensions.

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

We provide several different methods for selecting the next point to evaluate. All
methods in this version are based in generating candidate points by perturbing the
best solution found so far or in some cases just choose a random point. We also
provide the option of using many different strategies in the same experiment and
how to cycle between the different strategies. Each implementation of this object
is required to have the following attributes and methods

- Attributes:
    * proposed_points: Number of data points (integer)
- Required methods
    * init(start_sample, fhat, budget): This initializes the sampling strategy
      by providing the points that were evaluated in the experimental design phase, the
      response surface, and also provides  the evaluation budget.
    * remove_point(x): Removes point x from list of proposed_points if the evaluation
      crashed or was never carried out by the strategy. Returns True if the point was
      removed and False if the removal failed.
    * make_points(npts, xbest, sigma, subset=None, proj_fun=None): This is the method
      that proposes npts new evaluations to the strategy. It needs to know
      the number of points to propose, the best data point evaluated so far, the
      preferred sample radius of the strategy (w.r.t the unit box), the coordinates
      that the strategy wants to perturb, and a way to project points onto the feasible
      region.

We now list the different options and describe shortly how they work.

- **CandidateSRBF:**
    Generate perturbations around the best solution found so far
- **CandidateSRBF_INT:**
    Uses CandidateSRBF but only perturbs the integer variables
- **CandidateSRBF_CONT:**
    Uses CandidateSRBF but only perturbs the continuous variables
- **CandidateDYCORS:**
    Uses a DYCORS strategy which perturbs each coordinate with
    some iteration dependent probability. This probability is
    a monotonically decreasing function with the number of iteration.
- **CandidateDYCORS_CONT:**
    Uses CandidateDYCORS but only perturbs the continuous variables
- **CandidateDYCORS_INT:**
    Uses CandidateDYCORS but only perturbs the integer variables
- **CandidateDDS:**
    Uses the DDS strategy where only a few candidate points are generated
    and the one with the best surrogate prediction is picked for evaluation
- **CandidateDDS_CONT:**
    Uses CandidateDDS but only perturbs the continuous variables
- **CandidateDDS_INT:**
    Uses CandidateDDS but only perturbs the integer variables
- **CandidateUniform:**
    Chooses a new point uniformly from the box-constrained domain
- **CandidateUniform_CONT:**
    Given the best solution found so far the continuous variables are
    chosen uniformly from the box-constrained domain
- **CandidateUniform_INT:**
    Given the best solution found so far the integer variables are
    chosen uniformly from the box-constrained domain

The CandidateDYCORS algorithm is the bread-and-butter algorithm for any
problems with more than 5 dimensions whilst CandidateSRBF is recommended
for problems with only a few dimensions. It is sometimes efficient in mixed-integer
problems to perturb the integer and continuous variables separately and we
therefore provide such method for each of these algorithms. Finally, uniformly
choosing a new point has the advantage of creating diversity to avoid getting
stuck in a local minima. Each method needs an objective function object as
described in the previous section (the input name is data) and how many
perturbations should be generated around the best solution found so far
(the input name is numcand). Around 100 points per dimension, but no more
than 5000, is recommended. Next is an example on how to generate a multi-start
strategy that uses CandidateDYCORS, CandidateDYCORS_CONT,
CandidateDYCORS_INT, and CandidateUniform and that cycles evenly between
the methods i.e., the first point is generated using CandidateDYCORS, the
second using CandidateDYCORS_CONT and so on.

.. code-block:: python

    from pySOT import LinearMI, MultiSampling, CandidateDYCORS, \
                  CandidateDYCORS_CONT, CandidateDYCORS_INT, \
                  CandidateUniform

    data = LinearMI()  # Optimization problem
    sampling_methods = [CandidateDYCORS(data=data, numcand=100*data.dim),
                        CandidateDYCORS_CONT(data=data, numcand=100*data.dim),
                        CandidateDYCORS_INT(data=data, numcand=100*data.dim),
                        CandidateUniform(data=data, numcand=100*data.dim)]
    cycle = [0, 1, 2, 3]
    sampling_methods = MultiSampling(sampling_methods, cycle)
