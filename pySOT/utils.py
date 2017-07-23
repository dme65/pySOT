"""
.. module:: utils
   :synopsis: Help functions for pySOT

.. moduleauthor:: David Eriksson <dme65@cornell.edu>

:Module: utils
:Author: David Eriksson <dme65@cornell.edu>

"""

import numpy as np
from pySOT.experimental_design import SymmetricLatinHypercube


def to_unit_box(x, data):
    """Maps a set of points to the unit box

    :param x: Points to be mapped to the unit box, of size npts x dim
    :type x: numpy.array
    :param data: Optimization problem, needs to have attributes xlow and xup
    :type data: Object
    :return: Points mapped to the unit box
    :rtype: numpy.array
    """

    return (np.copy(x) - data.xlow) / (data.xup - data.xlow)


def from_unit_box(x, data):
    """Maps a set of points from the unit box to the original domain

    :param x: Points to be mapped from the unit box, of size npts x dim
    :type x: numpy.array
    :param data: Optimization problem, needs to have attributes xlow and xup
    :type data: Object
    :return: Points mapped to the original domain
    :rtype: numpy.array
    """

    return data.xlow + (data.xup - data.xlow) * np.copy(x)


def unit_rescale(xx):
    """Shift and rescale elements of a vector to the unit interval

    :param xx: Vector that should be rescaled to the unit interval
    :type xx: numpy.array
    :return: Vector scaled to the unit interval
    :rtype: numpy.array
    """

    xmax = np.amax(xx)
    xmin = np.amin(xx)
    if xmax == xmin:
        return np.ones(xx.shape)
    else:
        return (xx-xmin)/(xmax-xmin)


def round_vars(data, x):
    """Round integer variables to closest integer that is still in the domain

    :param data: Optimization problem object
    :type data: Object
    :param x: Set of points, of size npts x dim
    :type x: numpy.array
    :return: The set of points with the integer variables
        rounded to the closest integer in the domain
    :rtype: numpy.array
    """

    if len(data.integer) > 0:
        # Round the original ranged integer variables
        x[:, data.integer] = np.round(x[:, data.integer])
        # Make sure we don't violate the bound constraints
        for i in data.integer:
            ind = np.where(x[:, i] < data.xlow[i])
            x[ind, i] += 1
            ind = np.where(x[:, i] > data.xup[i])
            x[ind, i] -= 1
    return x


def check_opt_prob(obj):
    """Routine for checking that an implementation of the optimization problem
    follows the standard. This method checks everything, but can't make
    sure that the objective function and constraint methods return values
    of the correct type since this would involve actually evaluating the
    objective function which isn't feasible when the evaluations are
    expensive. If some test fails, an exception is raised through assert.

    :param obj: Optimization problem
    :type obj: Object
    :raise AttributeError: If object doesn't follow the pySOT standard
    """

    if not hasattr(obj, "dim"):
        raise AttributeError("Problem dimension required")
    if not hasattr(obj, "xlow"):
        raise AttributeError("Numpy array of lower bounds required")
    if not isinstance(obj.xlow, np.ndarray):
        raise AttributeError("Numpy array of lower bounds required")
    if not hasattr(obj, "xup"):
        raise AttributeError("Numpy array of upper bounds required")
    if not isinstance(obj.xup, np.ndarray):
        raise AttributeError("Numpy array of upper bounds required")
    if not hasattr(obj, "integer"):
        raise AttributeError("Integer variables must be specified")
    if len(obj.integer) > 0:
        if not isinstance(obj.integer, np.ndarray):
            raise AttributeError("Integer variables must be specified")
    else:
        if not(isinstance(obj.integer, np.ndarray) or
               isinstance(obj.integer, list)):
            raise AttributeError("Integer variables must be specified")
    if not hasattr(obj, "continuous"):
        raise AttributeError("Continuous variables must be specified")
    if len(obj.continuous) > 0:
        if not isinstance(obj.continuous, np.ndarray):
            raise AttributeError("Continuous variables must be specified")
    else:
        if not(isinstance(obj.continuous, np.ndarray) or
               isinstance(obj.continuous, list)):
            raise AttributeError("Continuous variables must be specified")

    # Check for logical errors
    if not (isinstance(obj.dim, int) and obj.dim > 0):
        raise AttributeError("Problem dimension must be a positive integer.")
    if not(len(obj.xlow) == obj.dim and
            len(obj.xup) == obj.dim):
        raise AttributeError("Incorrect size for xlow and xup")
    if not(all(obj.xlow[i] < obj.xup[i] for i in range(obj.dim))):
        raise AttributeError("Lower bounds must be below upper bounds.")
    if len(obj.integer) > 0:
        if not(np.amax(obj.integer) < obj.dim and np.amin(obj.integer) >= 0):
            raise AttributeError("Integer variable index can't exceed "
                                 "number of dimensions or be negative")
    if len(obj.continuous) > 0:
        if not(np.amax(obj.continuous) < obj.dim and
               np.amin(obj.continuous) >= 0):
            raise AttributeError("Continuous variable index can't exceed "
                                 "number of dimensions or be negative")
    if not(len(np.intersect1d(obj.continuous, obj.integer)) == 0):
        raise AttributeError("A variable can't be both an integer and continuous")
    if not(len(obj.continuous)+len(obj.integer) == obj.dim):
        raise AttributeError("All variables must be either integer or continuous")


def progress_plot(controller, title='', interactive=False):
    """Makes a progress plot from a POAP controller

    This method depends on matplotlib and will terminate if matplotlib.pyplot
    is unavailable.

    :param controller: POAP controller object
    :type controller: Object
    :param title: Title of the plot
    :type title: string
    :param interactive: True if the plot should be interactive
    :type interactive: bool
    """

    try:
        import matplotlib.pyplot as plt
        plotting_on = True
    except:
        plotting_on = False
        pass

    if not plotting_on:
        print("Failed to import matplotlib.pyplot, aborting....")
        return

    # Extract function values from the controller, ignoring crashed evaluations
    fvals = np.array([o.value for o in controller.fevals if o.value is not None])

    plt.figure()
    if interactive:
        plt.ion()
    plt.plot(np.arange(0, fvals.shape[0]), fvals, 'bo')  # Points
    plt.plot(np.arange(0, fvals.shape[0]), np.minimum.accumulate(fvals),
             'r-', linewidth=4.0)  # Best value found

    # Set limits
    ymin = np.min(fvals)
    ymax = np.max(fvals)
    plt.ylim(ymin - 0.1 * (ymax - ymin), ymax + 0.1 * (ymax - ymin))

    plt.xlabel('Evaluations')
    plt.ylabel('Function Value')
    plt.title(title)
    plt.show()


class RSCapped(object):
    """Cap adapter for response surfaces.

    This adapter takes an existing response surface and replaces it
    with a modified version in which the function values are replaced
    according to some transformation. A very common transformation
    is to replace all values above the median by the median in order
    to reduce the influence of large function values.

    :param model: Original response surface object
    :type model: Object
    :param transformation: Function value transformation object. Median capping
        is used if no object (or None) is provided
    :type transformation: Object

    :ivar transformation: Object used to transform the function values.
    :ivar model: original response surface object
    :ivar fvalues: Function values
    :ivar nump: Current number of points
    :ivar maxp: Initial maximum number of points (can grow)
    :ivar updated: True if the surface is updated
    """

    def __init__(self, model, transformation=None):

        self.transformation = transformation
        if self.transformation is None:
            def transformation(fvalues):
                medf = np.median(fvalues)
                fvalues[fvalues > medf] = medf
                return fvalues
            self.transformation = transformation
        self.model = model
        self.fvalues = np.zeros((model.maxp, 1))
        self.nump = 0
        self.maxp = model.maxp
        self.updated = True

    def reset(self):
        """Reset the capped response surface"""

        self.model.reset()
        self.fvalues[:] = 0
        self.nump = 0

    def add_point(self, xx, fx):
        """Add a new function evaluation

        :param xx: Point to add
        :type xx: numpy.array
        :param fx: The function value of the point to add
        :type fx: float
        """

        if self.nump >= self.fvalues.shape[0]:
            self.fvalues.resize(2*self.fvalues.shape[0], 1)
        self.fvalues[self.nump] = fx
        self.nump += 1
        self.updated = False
        self.model.add_point(xx, fx)

    def get_x(self):
        """Get the list of data points

        :return: List of data points
        :rtype: numpy.array
        """

        return self.model.get_x()

    def get_fx(self):
        """Get the list of function values for the data points.

        :return: List of function values
        :rtype: numpy.array
        """

        return self.model.get_fx()

    def eval(self, x, ds=None):
        """Evaluate the capped interpolant at the point x

        :param x: Point where to evaluate
        :type x: numpy.array
        :return: Value of the RBF interpolant at x
        :rtype: float
        """

        self._apply_transformation()
        return self.model.eval(x, ds)

    def evals(self, x, ds=None):
        """Evaluate the capped interpolant at the points x

        :param x: Points where to evaluate, of size npts x dim
        :type x: numpy.array
        :param ds: Distances between the centers and the points x, of size npts x ncenters
        :type ds: numpy.array
        :return: Values of the capped interpolant at x, of length npts
        :rtype: numpy.array
        """

        self._apply_transformation()
        return self.model.evals(x, ds)

    def deriv(self, x, ds=None):
        """Evaluate the derivative of the capped interpolant at a point x

        :param x: Point for which we want to compute the RBF gradient
        :type x: numpy.array
        :param ds: Distances between the centers and the point x
        :type ds: numpy.array
        :return: Derivative of the capped interpolant at x
        :rtype: numpy.array
        """

        self._apply_transformation()
        return self.model.deriv(x, ds)

    def _apply_transformation(self):
        """ Apply the cap to the function values."""

        fvalues = np.copy(self.fvalues[0:self.nump])
        self.model.transform_fx(self.transformation(fvalues))


class RSPenalty(object):
    """Penalty adapter for response surfaces.

    This adapter can be used for approximating an objective function plus
    a penalty function. The response surface is fitted only to the objective
    function and the penalty is added on after.

    :param model: Original response surface object
    :type model: Object
    :param evals: Object that takes the response surface and the points and adds up
        the response surface value and the penalty function value
    :type evals: Object
    :param devals: Object that takes the response surface and the points and adds up
        the response surface derivative and the penalty function derivative
    :type devals: Object

    :ivar eval_method: Object that takes the response surface and the points and adds up
        the response surface value and the penalty function value
    :ivar deval_method: Object that takes the response surface and the points and adds up
        the response surface derivative and the penalty function derivative
    :ivar model: original response surface object
    :ivar fvalues: Function values
    :ivar nump: Current number of points
    :ivar maxp: Initial maximum number of points (can grow)
    :ivar updated: True if the surface is updated
    """

    def __init__(self, model, evals, derivs):

        self.model = model
        self.fvalues = np.zeros((model.maxp, 1))
        self.nump = 0
        self.maxp = model.maxp
        self.eval_method = evals
        self.deriv_method = derivs
        self.updated = True

    def reset(self):
        """Reset the capped response surface"""

        self.model.reset()
        self.fvalues[:] = 0
        self.nump = 0

    def add_point(self, xx, fx):
        """Add a new function evaluation

        :param xx: Point to add
        :type xx: numpy.array
        :param fx: The function value of the point to add
        :type fx: float
        """

        if self.nump >= self.fvalues.shape[0]:
            self.fvalues.resize(2*self.fvalues.shape[0], 1)
        self.fvalues[self.nump] = fx
        self.nump += 1
        self.updated = False
        self.model.add_point(xx, fx)

    def get_x(self):
        """Get the list of data points

        :return: List of data points
        :rtype: numpy.array
        """

        return self.model.get_x()

    def get_fx(self):
        """Get the list of function values for the data points.

        :return: List of function values
        :rtype: numpy.array
        """

        return self.eval_method(self.model, self.model.get_x())[0, 0]

    def eval(self, x, ds=None):
        """Evaluate the penalty adapter interpolant at the point xx

        :param x: Point where to evaluate
        :type x: numpy.array
        :param ds: Not used
        :type ds: None
        :return: Value of the interpolant at x
        :rtype: float
        """

        return self.eval_method(self.model, np.atleast_2d(x)).ravel()

    def evals(self, x, ds=None):
        """Evaluate the penalty adapter at the points xx

        :param x: Points where to evaluate, of size npts x dim
        :type x: numpy.array
        :param ds: Not used
        :type ds: None
        :return: Values of the interpolant at x, of length npts
        :rtype: numpy.array
        """

        return self.eval_method(self.model, x)

    def deriv(self, x, ds=None):
        """Evaluate the derivative of the penalty adapter at x

        :param x: Point for which we want to compute the gradient
        :type x: numpy.array
        :param ds: Not used
        :type ds: None
        :return: Derivative of the interpolant at x
        :rtype: numpy.array
        """

        return self.deriv_method(self.model, x)


class RSUnitbox(object):
    """Unit box adapter for response surfaces

    This adapter takes an existing response surface and replaces it
    with a modified version where the domain is rescaled to the unit
    box. This is useful for response surfaces that are sensitive to
    scaling, such as radial basis functions.

    :param model: Original response surface object
    :type model: Object
    :param data: Optimization problem object
    :type data: Object

    :ivar data: Optimization problem object
    :ivar model: original response surface object
    :ivar fvalues: Function values
    :ivar nump: Current number of points
    :ivar maxp: Initial maximum number of points (can grow)
    :ivar updated: True if the surface is updated
    """

    def __init__(self, model, data):

        self.model = model
        self.fvalues = np.zeros((model.maxp, 1))
        self.nump = 0
        self.maxp = model.maxp
        self.data = data
        self.updated = True

    def reset(self):
        """Reset the capped response surface"""

        self.model.reset()
        self.fvalues[:] = 0
        self.nump = 0

    def add_point(self, xx, fx):
        """Add a new function evaluation

        :param xx: Point to add
        :type xx: numpy.array
        :param fx: The function value of the point to add
        :type fx: float
        """

        if self.nump >= self.fvalues.shape[0]:
            self.fvalues.resize(2*self.fvalues.shape[0], 1)
        self.fvalues[self.nump] = fx
        self.nump += 1
        self.updated = False
        self.model.add_point(to_unit_box(xx, self.data), fx)

    def get_x(self):
        """Get the list of data points

        :return: List of data points
        :rtype: numpy.array
        """

        return from_unit_box(self.model.get_x(), self.data)

    def get_fx(self):
        """Get the list of function values for the data points.

        :return: List of function values
        :rtype: numpy.array
        """

        return self.model.get_fx()

    def eval(self, x, ds=None):
        """Evaluate the response surface at the point xx

        :param x: Point where to evaluate
        :type x: numpy.array
        :param ds: Not used
        :type ds: None
        :return: Value of the interpolant at x
        :rtype: float
        """

        return self.model.eval(to_unit_box(x, self.data), ds)

    def evals(self, x, ds=None):
        """Evaluate the capped rbf interpolant at the points xx

        :param x: Points where to evaluate, of size npts x dim
        :type x: numpy.array
        :param ds: Not used
        :type ds: None
        :return: Values of the MARS interpolant at x, of length npts
        :rtype: numpy.array
        """

        return self.model.evals(to_unit_box(x, self.data), ds)

    def deriv(self, x, ds=None):
        """Evaluate the derivative of the rbf interpolant at x

        :param x: Point for which we want to compute the MARS gradient
        :type x: numpy.array
        :param ds: Not used
        :type ds: None
        :return: Derivative of the MARS interpolant at x
        :rtype: numpy.array
        """

        return self.model.deriv(to_unit_box(x, self.data), ds)


class GeneticAlgorithm:
    """Genetic algorithm

    This is an implementation of the real-valued Genetic algorithm that is useful for optimizing
    on a surrogate model, but it can also be used on its own. The mutations are normally distributed
    perturbations, the selection mechanism is a tournament selection, and the crossover oepration is
    the standard linear combination taken at a randomly generated cutting point.

    The number of evaluations are popsize x ngen

    :param function: Function that can be used to evaluate the entire population. It needs to
        take an input of size nindividuals x nvariables and return a numpy.array of length
        nindividuals
    :type function: Object
    :param dim: Number of dimensions
    :type dim: int
    :param xlow: Lower variable bounds, of length dim
    :type xlow: numpy.array
    :param xup: Lower variable bounds, of length dim
    :type xup: numpy.array
    :param intvar: List of indices with the integer valued variables (e.g., [0, 1, 5])
    :type intvar: list
    :param popsize: Population size
    :type popsize: int
    :param ngen: Number of generations
    :type ngen: int
    :param start: Method for generating the initial population
    :type start: string
    :param proj_fun: Function that can project ONE infeasible individual onto the feasible region
    :type proj_fun: Object

    :ivar nvariables: Number of variables (dimensions) of the objective function
    :ivar nindividuals: population size
    :ivar lower_boundary: lower bounds for the optimization problem
    :ivar upper_boundary: upper bounds for the optimization problem
    :ivar integer_variables: List of variables that are integer valued
    :ivar start: Method for generating the initial population
    :ivar sigma: Perturbation radius. Each pertubation is N(0, sigma)
    :ivar p_mutation: Mutation probability (1/dim)
    :ivar tournament_size: Size of the tournament (5)
    :ivar p_cross: Cross-over probability (0.9)
    :ivar ngenerations: Number of generations
    :ivar function: Object that can be used to evaluate the objective function
    :ivar projfun: Function that can be used to project an individual onto the feasible region
    """

    def __init__(self, function, dim, xlow, xup, intvar=None, popsize=100, ngen=100, start="SLHD", projfun=None):
        self.nvariables = dim
        self.nindividuals = popsize + (popsize % 2)  # Make sure this is even
        self.lower_boundary = np.array(xlow)
        self.upper_boundary = np.array(xup)
        self.integer_variables = []
        if intvar is not None:
            self.integer_variables = np.array(intvar)
        self.start = start
        self.sigma = 0.2
        self.p_mutation = 1.0/dim
        self.tournament_size = 5
        self.p_cross = 0.9
        self.ngenerations = ngen
        self.function = function
        self.projfun = projfun

    def optimize(self):
        """Method used to run the Genetic algorithm

        :return: Returns the best individual and its function value
        :rtype: numpy.array, float
        """

        #  Initialize population
        if isinstance(self.start, np.ndarray):
            if self.start.shape[0] != self.nindividuals or self.start.shape[1] != self.nvariables:
                raise ValueError("Unknown method for generating the initial population")
            if (not all(np.min(self.start, axis=0) >= self.lower_boundary)) or \
                    (not all(np.max(self.start, axis=0) <= self.upper_boundary)):
                raise ValueError("Initial population is outside the domain")
            population = self.start
        elif self.start == "SLHD":
            exp_des = SymmetricLatinHypercube(self.nvariables, self.nindividuals)
            population = self.lower_boundary + exp_des.generate_points() * \
                (self.upper_boundary - self.lower_boundary)
        elif self.start == "LHD":
            exp_des = LatinHypercube(self.nvariables, self.nindividuals)
            population = self.lower_boundary + exp_des.generate_points() * \
                (self.upper_boundary - self.lower_boundary)
        elif self.start == "Random":
            population = self.lower_boundary + np.random.rand(self.nindividuals, self.nvariables) *\
                (self.upper_boundary - self.lower_boundary)
        else:
            raise ValueError("Unknown argument for initial population")

        new_population = []
        #  Round positions
        if len(self.integer_variables) > 0:
            new_population = np.copy(population)
            population[:, self.integer_variables] = np.round(population[:, self.integer_variables])
            for i in self.integer_variables:
                ind = np.where(population[:, i] < self.lower_boundary[i])
                population[ind, i] += 1
                ind = np.where(population[:, i] > self.upper_boundary[i])
                population[ind, i] -= 1

        #  Evaluate all individuals
        function_values = self.function(population)
        if len(function_values.shape) == 2:
            function_values = np.squeeze(np.asarray(function_values))

        # Save the best individual
        ind = np.argmin(function_values)
        best_individual = np.copy(population[ind, :])
        best_value = function_values[ind]

        if len(self.integer_variables) > 0:
            population = new_population

        # Main loop
        for ngen in range(self.ngenerations):
            # Do tournament selection to select the parents
            competitors = np.random.randint(0, self.nindividuals, (self.nindividuals, self.tournament_size))
            ind = np.argmin(function_values[competitors], axis=1)
            winner_indices = np.zeros(self.nindividuals, dtype=int)
            for i in range(self.tournament_size):  # This loop is short
                winner_indices[np.where(ind == i)] = competitors[np.where(ind == i), i]

            parent1 = population[winner_indices[0:self.nindividuals//2], :]
            parent2 = population[winner_indices[self.nindividuals//2:self.nindividuals], :]

            # Averaging Crossover
            cross = np.where(np.random.rand(self.nindividuals//2) < self.p_cross)[0]
            nn = len(cross)  # Number of crossovers
            alpha = np.random.rand(nn, 1)

            # Create the new chromosomes
            parent1_new = np.multiply(alpha, parent1[cross, :]) + np.multiply(1-alpha, parent2[cross, :])
            parent2_new = np.multiply(alpha, parent2[cross, :]) + np.multiply(1-alpha, parent1[cross, :])
            parent1[cross, :] = parent1_new
            parent2[cross, :] = parent2_new
            population = np.concatenate((parent1, parent2))

            # Apply mutation
            scale_factors = self.sigma * (self.upper_boundary - self.lower_boundary)  # Account for dimensions ranges
            perturbation = np.random.randn(self.nindividuals, self.nvariables)  # Generate perturbations
            perturbation = np.multiply(perturbation, scale_factors)  # Scale accordingly
            perturbation = np.multiply(perturbation, (np.random.rand(self.nindividuals,
                                                                     self.nvariables) < self.p_mutation))

            population += perturbation  # Add perturbation
            population = np.maximum(np.reshape(self.lower_boundary, (1, self.nvariables)), population)
            population = np.minimum(np.reshape(self.upper_boundary, (1, self.nvariables)), population)

            # Map to feasible region if method exists
            if self.projfun is not None:
                for i in range(self.nindividuals):
                    population[i, :] = self.projfun(population[i, :])

            # Round chromosomes
            new_population = []
            if len(self.integer_variables) > 0:
                new_population = np.copy(population)
                population[:, self.integer_variables] = np.round(population[:, self.integer_variables])
                for i in self.integer_variables:
                    ind = np.where(population[:, i] < self.lower_boundary[i])
                    population[ind, i] += 1
                    ind = np.where(population[:, i] > self.upper_boundary[i])
                    population[ind, i] -= 1

            # Keep the best individual
            population[0, :] = best_individual

            #  Evaluate all individuals
            function_values = self.function(population)
            if len(function_values.shape) == 2:
                function_values = np.squeeze(np.asarray(function_values))

            # Save the best individual
            ind = np.argmin(function_values)
            best_individual = np.copy(population[ind, :])
            best_value = function_values[ind]

            # Use the positions that are not rounded
            if len(self.integer_variables) > 0:
                population = new_population

        return best_individual, best_value
