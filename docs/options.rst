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

The following strategies are currently supported:

SRBFStrategy
^^^^^^^^^^^^
This is an implementation of the SRBF strategy by Regis and Shoemaker:

| Rommel G Regis and Christine A Shoemaker.
| A stochastic radial basis function method for the global optimization of expensive functions.
| INFORMS Journal on Computing, 19(4): 497–509, 2007.

Rommel G Regis and Christine A Shoemaker.
Parallel stochastic global optimization using radial basis functions.
INFORMS Journal on Computing, 21(3):411–426, 2009.

The main idea is to pick the new evaluations from a set of candidate
points where each candidate point is generated as an N(0, sigma^2)
distributed perturbation from the current best solution. The value of
sigma is modified based on progress and follows the same logic as in many
trust region methods; we increase sigma if we make a lot of progress
(the surrogate is accurate) and decrease sigma when we aren't able to
make progress (the surrogate model is inaccurate). More details about how
sigma is updated is given in the original papers.

After generating the candidate points we predict their objective function
value and compute the minimum distance to previously evaluated point. Let
the candidate points be denoted by C and let the function value predictions
be s(x_i) and the distance values be d(x_i), both rescaled through a linear
transformation to the interval [0,1]. This is done to put the values on the
same scale. The next point selected for evaluation is the candidate point
x that minimizes the weighted-distance merit function:

.. math::

    \text{merit}(x) := w s(x) + (1 - w) (1 - d(x))

where :math:`0 \leq w \leq 1`. That is, we want a small function value prediction and a
large minimum distance from previously evalauted points. The weight w is
commonly cycled between a few values to achieve both exploitation and
exploration. When w is close to zero we do pure exploration while w close
to 1 corresponds to explotation.

- Parameters:
    * max_evals: Evaluation budget (int)
    * opt_prob: Optimization problem object, must implement OptimizationProblem
    * exp_design: Experimental design object, must implement ExperimentalDesign
    * surrogate: Surrogate object, must implement Surrogate
    * asynchronous: Whether or not to use asynchrony (True / False).
    * batch_size: Size of the batch. This value is ignored if asynchronous is True. Use 1 for serial or run with asynchronous set to True.
    * extra_points: n Extra points to add to the experimental design (numpy.array of size n x dim)
    * extra_vals: Values for extra_points. Set elements to np.nan if unknown (numpy.array of size n x 1)
    * reset_surrogate: Specify whether or not we are resetting the surrogate model i.e., removing current points (True / False)
    * weights: Weights for merit function (list or numpy.array). Default is [0.3, 0.5, 0.8, 0.95]
    * num_cand: Number of candidate points (int). Default = 100*dim

DYCORStrategy
^^^^^^^^^^^^^

This is an implementation of the DYCORS strategy by Regis and Shoemaker:

| Rommel G Regis and Christine A Shoemaker.
| Combining radial basis function surrogates and dynamic coordinate search in
  high-dimensional expensive black-box optimization.
| Engineering Optimization, 45(5): 529–555, 2013.

This is an extension of the SRBF strategy that changes how the candidate
points are generated. The main idea is that many objective functions depend
only on a few directions so it may be advantageous to perturb only a few
directions. In particular, we use a perturbation probability to perturb a
given coordinate and decrease this probability after each function
evaluation so fewer coordinates are perturbed later in the optimization.

The parameters are the same as in the SRBF strategy.

SOPStrategy
^^^^^^^^^^^

This is an implementation of the SOP strategy by Krityakierne, Akhtar and Shoemaker:

| Tipaluck Krityakierne, Taimoor Akhtar and Christine A. Shoemaker.
| SOP: parallel surrogate global optimization with Pareto center selection
  for computationally expensive single objective problems.
| Journal of Global Optimization, 66(3): 417–437, 2016.

The core idea of SOP is to maintain a ranked archive of all previously evaluated points,
as per non-dominated sorting between two objectives, i.e., i) Objective function value(minimize)
and ii) Minimum distance from other evaluated points(maximize). A sub-archive of center points
is subsequently maintained via selection from the ranked evaluated points. The number of points
in the sub-archive of centers should be equal to (or greater than) the number of parallel threads.
Candidate points are generated around each ‘center point’ via the DYCORS sampling strategy, i.e.,
an N(0, sigma^2) distributed perturbation of a subset of decision variables. A separate value of
sigma is maintained for each center point, where  sigma is decreased if no progress is registered
in the bi-objective objective value and distance criterion trade-off. One point is selected for
expensive evaluation from each set of candidate points, based on the surrogate approximation only.
Hence the merit function is s(x), where s(x) is the surrogate prediction.

Exploration and exploitation are simultaneously achieved (in parallel) via the bi-objective ranking
of previously evaluated points, and subsequent selection of these points as centers of DYCORS perturbations.
Exploitation is achieved when the point with best objective value is the perturbation center, and the
candidate around it with best surrogate value is selected as the new evaluation point. Exploration is
achieved when the point with the maximum distance (max-min) from other evaluated points is selected as
the perturbation center.

Parameters are the same as in SRBF strategy, but exclude weights, and include the following:
 - ncenters: Number of center points for candidate search where one point is selected for evaluation per, each center.

EIStrategy
^^^^^^^^^^

This is an implementation of Expected Improvement (EI), arguably the most
popular acquisition function in Bayesian optimization. Under a Gaussian
process (GP) prior, the expected value of the improvement:

.. math::
    \begin{align*}
        \text{I}(x) &:= \max(f_{\text{best}} - f(x), 0) \\
        \text{EI}[x] &:= \mathbb{E}[I(x)]
    \end{align*}

can be computed analytically, where f_best is the best observed function
value.EI is one-step optimal in the sense that selecting the maximizer of
EI is the optimal action if we have exactly one function value remaining
and must return a solution with a known function value.

When using parallelism, we constrain each new evaluation to be a distance
dtol away from previous and pending evaluations to avoid that the same
point is being evaluated multiple times. We use a default value of
dtol = 1e-3 * norm(ub - lb), but note that this value has not been
tuned carefully and may be far from optimal.

The optimization strategy terminates when the evaluatio budget has been
exceeded or when the EI of the next point falls below some threshold,
where the default threshold is 1e-6 * (max(fX) -  min(fX)).

- Parameters:
    * max_evals: Evaluation budget (int)
    * opt_prob: Optimization problem object, must implement OptimizationProblem
    * exp_design: Experimental design object, must implement ExperimentalDesign
    * surrogate: Surrogate object, must implement Surrogate
    * asynchronous: Whether or not to use asynchrony (True / False).
    * batch_size: Size of the batch. This value is ignored if asynchronous is True. Use 1 for serial or run with asynchronous set to True.
    * extra_points: n Extra points to add to the experimental design (numpy.array of size n x dim)
    * extra_vals: Values for extra_points. Set elements to np.nan if unknown (numpy.array of size n x 1)
    * reset_surrogate: Specify whether or not we are resetting the surrogate model i.e., removing current points (True / False)
    * ei_tol: Terminate if the largest EI falls below this threshold (float). Default: 1e-6 * (max(fX) -  min(fX))
    * dtol: Minimum distance between new and pending/finished evaluations (float). Default: 1e-3 * norm(ub - lb)


LCBStrategy
^^^^^^^^^^^

This is an implementation of Lower Confidence Bound (LCB), a
popular acquisition function in Bayesian optimization. The main idea
is to minimize:

.. math::
    \text{LCB}(x) := \mathbb{E}[x] - \kappa * \sqrt{\mathbb{V}[x]}

where :math:`\mathbb{E}[x]` is the predicted function value, :math:`V[x]` is the predicted
variance, and kappa is a constant that balances exploration and
exploitation. We use a default value of kappa = 2.

When using parallelism, we constrain each new evaluation to be a distance
dtol away from previous and pending evaluations to avoid that the same
point is being evaluated multiple times. We use a default value of
dtol = 1e-3 * norm(ub - lb), but note that this value has not been
tuned carefully and may be far from optimal.

The optimization strategy terminates when the evaluatio budget has been
exceeded or when the LCB of the next point falls below some threshold,
where the default threshold is 1e-6 * (max(fX) -  min(fX)).

- Parameters:
    * max_evals: Evaluation budget (int)
    * opt_prob: Optimization problem object, must implement OptimizationProblem
    * exp_design: Experimental design object, must implement ExperimentalDesign
    * surrogate: Surrogate object, must implement Surrogate
    * asynchronous: Whether or not to use asynchrony (True / False).
    * batch_size: Size of the batch. This value is ignored if asynchronous is True. Use 1 for serial or run with asynchronous set to True.
    * extra_points: n Extra points to add to the experimental design (numpy.array of size n x dim)
    * extra_vals: Values for extra_points. Set elements to np.nan if unknown (numpy.array of size n x 1)
    * reset_surrogate: Specify whether or not we are resetting the surrogate model i.e., removing current points (True / False)
    * kappa: Constant in the LCB merit function (float). Default: 2.0
    * lcb_tol: Terminate if min(fX) - min(LCB(x)) < lcb_tol (float). Default: 1e-6 * (max(fX) -  min(fX))
    * dtol: Minimum distance between new and pending/finished evaluations (float). Default: 1e-3 * norm(ub - lb)


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
    * generate_points(lb, ub, int_var): Returns an experimental design of size num_pts x dim where
      num_pts is the number of points in the initial design, which was specified
      when the object was created. You can supply lb, ub, and int_var to have the design mapped
      before it's scored instead of having the rounding take place in the strategy.

The following experimental designs are supported:

LatinHypercube
^^^^^^^^^^^^^^

A Latin hypercube design

- Parameters:
    * dim: Number of dimensions (int).
    * num_pts: Number of desired sampling points (int).
    * iterations: Number of designs to generate and choose the best from (int)

Example:

.. code-block:: python

    from pySOT import LatinHypercube
    exp_des = LatinHypercube(dim=3, num_pts=10)

creates a Latin hypercube design with 10 points in 3 dimensions

SymmetricLatinHypercube
^^^^^^^^^^^^^^^^^^^^^^^

A symmetric Latin hypercube design

- Parameters:
    * dim: Number of dimensions (int).
    * num_pts: Number of desired sampling points (int). Use 2*dim + 1 to make sure the design has full rank.
    * iterations: Number of designs to generate and choose the best from (int)

Example:

.. code-block:: python

    from pySOT import SymmetricLatinHypercube
    exp_des = SymmetricLatinHypercube(dim=3, num_pts=10)

creates a symmetric Latin hypercube design with 10 points in 3 dimensions

TwoFactorial
^^^^^^^^^^^^

The corners of the unit hypercube

- Parameters:
    * dim: Number of dimensions (int).

Example:

.. code-block:: python

    from pySOT import TwoFactorial
    exp_des = TwoFactorial(dim=3)

creates a two factorial design with 8 points in 3 dimensions

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
    * add_points(x, fx): Adds point(s) x with value(s) fx to the surrogate model. This **SHOULD NOT** trigger a new fit of the model.
    * predict(x): Evaluates the surrogate model at points x
    * predict_deriv(x): Evaluates the derivative of surrogate model at points x
- Optional methods
    * predict_std(x): Evaluates the uncertainty of the surrogate model at points x

The following surrogate models are supported:

RBFInterpolant
^^^^^^^^^^^^^^

A radial basis function (RBF) takes the form:

.. math:: s(x) = \sum_j c_j \phi(\|x-x_j\|) + \sum_j \lambda_j p_j(x)

where the functions :math:`p_j(x)` are low-degree polynomials.
The fitting equations are

.. math::
    \begin{bmatrix} \eta I & P^T \\ P & \Phi + \eta I \end{bmatrix}
    \begin{bmatrix} \lambda \\ c \end{bmatrix} =
    \begin{bmatrix} 0 \\ f \end{bmatrix}

where :math:`P_{ij} = p_j(x_i)` and :math:`\Phi_{ij}=\phi(\|x_i-x_j\|)`
The regularization parameter :math:`\eta` allows us to avoid problems
with potential poor conditioning of the system. Consider using the
SurrogateUnitBox wrapper or manually scaling the domain to the unit
hypercube to avoid issues with the domain scaling.

We add k new points to the RBFInterpolant in :math:`O(kn^2)` flops by
updating the LU factorization of the old RBF system. This is better than
computing the RBF coefficients from scratch, which costs :math:`O(n^3)` flops.

- Parameters:
    * dim: Number of dimensions (int)
    * kernel: RBF kernel object, must implement Kernel. Default: CubicKernel()
    * tail: RBF polynomial tail object, must implement Tail. Default: LinearTail(dim)
    * eta: Regularization parameter. Use something small like 1e-6 if the domain is [0, 1]^dim

Example:

.. code-block:: python

    from pySOT.surrogate import RBFInterpolant, CubicKernel, LinearTail
    fhat = RBFInterpolant(dim=dim, kernel=CubicKernel(), tail=LinearTail(dim=dim))

creates a cubic RBF with a linear tail in dim dimensions.

GPRegressor
^^^^^^^^^^^

Generate a Gaussian process regression object. This is just a wrapper around the GPRegressor in scikit-learn.

- Parameters:
    * dim: Number of dimensions (int)
    * gp: GPRegressor model in scikit-learn. Uses the SE/RBF/Gaussian kernel as a default if None is passed.
    * n_restarts_optimizer: Number of restarts in hyperparamater fitting (int)

Example:

.. code-block:: python

    from pySOT.surrogate import GPRegressor
    surrogate = GPRegressor(dim=dim)

creates a GPRegressor object in dim dimensions.

MARSInterpolant
^^^^^^^^^^^^^^^

Generate a Multivariate Adaptive Regression Splines (MARS) model.

.. math::

    \hat{f}(x) = \sum_{i=1}^{k} c_i B_i(x).

The model is a weighted sum of basis functions :math:`B_i(x)`. Each basis
function :math:`B_i(x)` takes one of the following three forms:

1. A constant 1.
2. A hinge function of the form :math:`\max(0, x - const)` or :math:`\max(0, const - x)`. MARS automatically selects variables and values of those variables for knots of the hinge functions.
3. A product of two or more hinge functions. These basis functions c an model interaction between two or more variables.

- Parameters:
    * dim: Number of dimensions (int)

.. note:: This implementation depends on the py-earth module (see :ref:`quickstart-label`)

Example:

.. code-block:: python

    from pySOT.surrogate import MARSInterpolant
    surrogate = MARSInterpolant(dim=dim)

creates a MARS interpolant in dim dimensions.

PolyRegressor
^^^^^^^^^^^^^

Multivariate polynomial regression with cross-terms. This is just a wrapper around PolynomialFeatures in scikit-learn.

- Parameters:
    * dim: Number of dimensions (int)
    * degree: Polynomial degree (int)

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
