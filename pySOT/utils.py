"""
.. module:: utils
   :synopsis: Help functions for pySOT

.. moduleauthor:: David Eriksson <dme65@cornell.edu>

:Module: utils
:Author: David Eriksson <dme65@cornell.edu>
"""

from pySOT.optimization_problems import OptimizationProblem
import numpy as np
import scipy.spatial as scp

POSITIVE_INFINITY = float("inf")


def to_unit_box(x, lb, ub):
    """Maps a set of points to the unit box

    :param x: Points to be mapped to the unit box, of size npts x dim
    :type x: numpy.array
    :param lb: Lower bounds, of size 1 x dim
    :type lb: numpy.array
    :param ub: Upper bounds, of size 1 x dim
    :type ub: numpy.array
    :return: Points mapped to the unit box
    :rtype: numpy.array
    """
    return (x - lb) / (ub - lb)


def from_unit_box(x, lb, ub):
    """Maps a set of points from the unit box to the original domain

    :param x: Points to be mapped from the unit box, of size npts x dim
    :type x: numpy.array
    :param lb: Lower bounds, of size 1 x dim
    :type lb: numpy.array
    :param ub: Upper bounds, of size 1 x dim
    :type ub: numpy.array
    :return: Points mapped to the original domain
    :rtype: numpy.array
    """
    return lb + (ub - lb) * x


def unit_rescale(x):
    """Shift and rescale elements of a vector to the unit interval

    :param x: array that should be rescaled to the unit interval
    :type x: numpy.ndarray
    :return: array scaled to the unit interval
    :rtype: numpy.ndarray
    """

    x_max = x.max()
    x_min = x.min()
    if x_max == x_min:
        return np.ones(x.shape)
    else:
        return (x - x_min)/(x_max - x_min)


def round_vars(x, int_var, lb, ub):
    """Round integer variables to closest integer in the domain.

    :param x: Set of points, of size npts x dim
    :type x: numpy.array
    :param int_var: Set of indices of integer variables
    :type int_var: numpy.array
    :param lb: Lower bounds, of size 1 x dim
    :type lb: numpy.array
    :param ub: Upper bounds, of size 1 x dim
    :type ub: numpy.array
    :return: The set of points with the integer variables
        rounded to the closest integer in the domain
    :rtype: numpy.array
    """

    if len(int_var) > 0:
        # Round the original ranged integer variables
        x[:, int_var] = np.round(x[:, int_var])
        # Make sure we don't violate the bound constraints
        for i in int_var:
            ind = np.where(x[:, i] < lb[i])
            x[ind, i] = lb[i]
            ind = np.where(x[:, i] > ub[i])
            x[ind, i] = ub[i]
    return x


def check_opt_prob(obj):  # pragma: no cover
    """Check an implementation of the optimization problem.

    This method checks everything, but can't make sure that the objective
    function returns values of the correct type since this would involve
    actually evaluating the objective function, which isn't feasible when
    the evaluations are expensive. If some test fails, an exception is raised.

    :param obj: Optimization problem
    :type obj: Object

    :raise Appropriate error if object doesn't follow the pySOT standard
    """
    if not isinstance(obj, OptimizationProblem):
        raise TypeError("Input does not implement OptimizationProblem")

    if not isinstance(obj.dim, int) and obj.dim() > 0:
        raise ValueError("dim must be a positive integer")

    if not isinstance(obj.lb, np.ndarray):
        raise ValueError("Numpy array of lower bounds required")
    if not isinstance(obj.ub, np.ndarray):
        raise ValueError("Numpy array of upper bounds required")
    if not(len(obj.lb) == obj.dim and len(obj.ub) == obj.dim):
        raise AttributeError("Incorrect size for lb and ub")
    if not(all(obj.lb[i] < obj.ub[i] for i in range(obj.dim))):
        raise AttributeError("Lower bounds must be smaller than upper bounds.")

    contvar = obj.cont_var
    intvar = obj.int_var
    if len(contvar) > 0:
        if not isinstance(contvar, np.ndarray):
            raise ValueError("Numpy array of continuous variables required")
        else:
            if not(isinstance(contvar, np.ndarray) or
                   isinstance(contvar, list)):
                raise AttributeError("Continuous variables must be specified")
    if len(intvar) > 0:
        if not isinstance(intvar, np.ndarray):
            raise ValueError("Numpy array of integer variables required")
        else:
            if not (isinstance(intvar, np.ndarray) or
                    isinstance(intvar, list)):
                raise AttributeError("Integer variables must be specified")

    if len(contvar) > 0:
        if not(np.amax(contvar) < obj.dim and np.amin(contvar) >= 0):
            raise AttributeError("Continuous variable index can't exceed "
                                 "number of dimensions or be negative")
    if len(intvar) > 0:
        if not(np.amax(intvar) < obj.dim and np.amin(intvar) >= 0):
            raise AttributeError("Integer variable index can't exceed "
                                 "number of dimensions or be negative")
    if not(len(np.intersect1d(contvar, intvar)) == 0):
        raise AttributeError("A variable can't be both "
                             "an integer and continuous")
    if not(len(contvar) + len(intvar) == obj.dim):
        raise AttributeError("All variables must be either "
                             "integer or continuous")


def progress_plot(controller, title="", interactive=False):  # pragma: no cover
    """Makes a progress plot from a POAP controller.

    This method requires matplotlib and will terminate if matplotlib.pyplot
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
    finally:
        if 'matplotlib.pyplot' not in sys.modules:
            plotting_on = False
            pass
        else:
            plotting_on = True

    if not plotting_on:
        print("Failed to import matplotlib.pyplot, aborting....")
        return

    # Extract function values from the controller, ignoring crashed evaluations
    fvals = np.array(
        [o.value for o in controller.fevals if o.value is not None])

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


class GeneticAlgorithm:
    """Genetic algorithm.

    Implementation of the real-valued Genetic algorithm. The mutations are
    normally distributed perturbations, the selection mechanism is a tournament
    selection, and the crossover oepration is the standard linear combination
    taken at a randomly generated cutting point.

    The total number of evaluations are popsize x ngen

    :param function: Function that can be used to evaluate the entire
        population. It needs to take an input of size pop_size x dim and
        return a numpy.array of size pop_size x 1
    :type function: Object
    :param dim: Number of dimensions
    :type dim: int
    :param lb: Lower variable bounds, of length dim
    :type lb: numpy.array
    :param ub: Lower variable bounds, of length dim
    :type ub: numpy.array
    :param int_var: List of indices with the integer valued variables
        (e.g., [0, 1, 5])
    :type int_var: list
    :param pop_size: Population size
    :type pop_size: int
    :param num_gen: Number of generations
    :type num_gen: int
    :param start: Method for generating the initial population
    :type start: string

    :ivar nvariables: Number of variables (dimensions)
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
    """

    def __init__(self, function, dim, lb, ub, int_var=None,
                 pop_size=100, num_gen=100, start="SLHD"):

        self.nvariables = dim
        self.nindividuals = pop_size + (pop_size % 2)  # Make sure this is even
        self.lower_boundary = np.array(lb)
        self.upper_boundary = np.array(ub)
        self.integer_variables = []
        if int_var is not None:
            self.integer_variables = np.array(int_var)
        self.start = start
        self.sigma = 0.2
        self.p_mutation = 1.0/dim
        self.tournament_size = 5
        self.p_cross = 0.9
        self.ngenerations = num_gen
        self.function = function

    def optimize(self):
        """Method used to run the Genetic algorithm

        :return: Returns the best individual and its function value
        :rtype: numpy.array, float
        """
        #  Initialize population
        if isinstance(self.start, np.ndarray):
            if self.start.shape[0] != self.nindividuals or \
                    self.start.shape[1] != self.nvariables:
                raise ValueError("Initial population has incorrect size")
            if any(np.min(self.start, axis=0) >= self.lower_boundary) or \
                    any(np.max(self.start, axis=0) <= self.upper_boundary):
                raise ValueError("Initial population is outside the domain")
            population = self.start
        elif self.start == "SLHD":
            from pySOT.experimental_design import SymmetricLatinHypercube
            exp_des = SymmetricLatinHypercube(
                self.nvariables, self.nindividuals)
            population = self.lower_boundary + exp_des.generate_points() * \
                (self.upper_boundary - self.lower_boundary)
        elif self.start == "LHD":
            from pySOT.experimental_design import LatinHypercube
            exp_des = LatinHypercube(self.nvariables, self.nindividuals)
            population = self.lower_boundary + exp_des.generate_points() * \
                (self.upper_boundary - self.lower_boundary)
        elif self.start == "Random":
            population = self.lower_boundary + np.random.rand(
                self.nindividuals, self.nvariables) *\
                (self.upper_boundary - self.lower_boundary)
        else:
            raise ValueError("Unknown argument for initial population")

        new_population = []
        #  Round positions
        if len(self.integer_variables) > 0:
            new_population = np.copy(population)
            population[:, self.integer_variables] = np.round(
                population[:, self.integer_variables])
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
        for _ in range(self.ngenerations):
            # Do tournament selection to select the parents
            competitors = np.random.randint(
                0, self.nindividuals,
                (self.nindividuals, self.tournament_size))
            ind = np.argmin(function_values[competitors], axis=1)
            winner_indices = np.zeros(self.nindividuals, dtype=int)
            for i in range(self.tournament_size):  # This loop is short
                winner_indices[np.where(ind == i)] = \
                    competitors[np.where(ind == i), i]

            parent1 = population[
                winner_indices[0:self.nindividuals//2], :]
            parent2 = population[
                winner_indices[self.nindividuals//2:self.nindividuals], :]

            # Averaging Crossover
            cross = np.where(np.random.rand(
                self.nindividuals//2) < self.p_cross)[0]
            nn = len(cross)  # Number of crossovers
            alpha = np.random.rand(nn, 1)

            # Create the new chromosomes
            parent1_new = np.multiply(alpha, parent1[cross, :]) + \
                np.multiply(1 - alpha, parent2[cross, :])
            parent2_new = np.multiply(alpha, parent2[cross, :]) + \
                np.multiply(1 - alpha, parent1[cross, :])
            parent1[cross, :] = parent1_new
            parent2[cross, :] = parent2_new
            population = np.concatenate((parent1, parent2))

            # Apply mutation
            scale_factors = self.sigma * (
                self.upper_boundary - self.lower_boundary)  # Scale
            perturbation = np.random.randn(
                self.nindividuals, self.nvariables)  # Generate perturbations
            perturbation = np.multiply(
                perturbation, scale_factors)  # Scale accordingly
            perturbation = np.multiply(perturbation, (
                np.random.rand(self.nindividuals,
                               self.nvariables) < self.p_mutation))

            population += perturbation  # Add perturbation
            population = np.maximum(np.reshape(
                self.lower_boundary, (1, self.nvariables)), population)
            population = np.minimum(np.reshape(
                self.upper_boundary, (1, self.nvariables)), population)

            # Round chromosomes
            new_population = []
            if len(self.integer_variables) > 0:
                new_population = np.copy(population)
                population = round_vars(population, self.integer_variables,
                                        self.lower_boundary,
                                        self.upper_boundary)

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


def domination(fA, fB):
    """Returns True if fA dominates fB

    :param fA: Vector A of objective functions
    :type fA: numy array of length M
    :param fB: Vector B of objective functions
    :type fB: numpy array of length M
    :return: Boolean check for domination
    :rtype: Bool
    """

    return np.all(np.less(fA, fB))


def nd_add(F, df_index, ndf_index):
    """Check if last row of F is dominated or non-dominated

    Updates the index vectors of domination and non-domination after adding
    the lastrow of the 2-D array F into the set of points being evaluated
    for non-domination

    :param F: 2D numpy array of objective function vectors
    :type F: numpy array
    :param df_index: Index array (int) of dominated vectors of F
    :type df_index: list of integers
    :param ndf_index: Index array (int) of non-dominated vectors of F
    :type ndf_index: list of integers
    :return: updated df_index and ndf_index
    :rtype: lists of integers
    """

    # Initialize and assume new point is non-dominated (nd)
    (M, N) = F.shape
    N = int(N - 1)
    ndf_count = len(ndf_index)
    ndf_index.append(N)
    ndf_count += 1
    j = 1

    # Traverse through F to update indices of dominated and nd points
    while j < ndf_count:
        if domination(F[:, N], F[:, ndf_index[j-1]]):
            df_index.append(ndf_index[j-1])
            ndf_index.remove(ndf_index[j-1])
            ndf_count -= 1
        elif domination(F[:, ndf_index[j-1]], F[:, N]):
            df_index.append(N)
            ndf_index.remove(N)
            ndf_count -= 1
            break
        else:
            j += 1
    return (ndf_index, df_index)


def nd_front(F):
    """ Find indices of dominated and nd points in F

    :param F: 2D array (MxN) containing a set of N vectors of M objectives
    :type F: 2D numpy array
    :return: 1) ndf_index --> indices of nd points in F,
             2) df_index --> indices of dominated points in F
    :rtype: lists of integers
    """

    (M, N) = F.shape
    df_index = []
    ndf_index = [int(0)]
    for i in range(1, N):
        (ndf_index, df_index) = nd_add(F[:, 0:i+1], df_index, ndf_index)
    return (ndf_index, df_index)


def nd_sorting(F, nmax):
    """ Computes the non-domination rank of each row vector in F

    This function sorts the objective functions vectors according to
    non-domination rank. If F is too large, sorting is only done for
    the first nmax points

    :param F: 2D array (MxN) containing a set of N vectors of M objectives
    :type F: numpy array
    :param nmax: maximum number of points ranked via nd_sorting
    :type nmax: int
    :return: list of N integers indicating rank of each row vector in F
    :rtype: list of size N
    """

    (M, N) = F.shape
    nd_ranks = np.ones((N, ), dtype=np.int)*POSITIVE_INFINITY
    P = np.ones((N, ), dtype=np.int)
    for i in range(0, N):
        P[i] = i
    i = 1
    count = 0
    while count < nmax and len(P) > 0:
        (ndf_index, df_index) = nd_front(F[:, P])
        for j in range(0, len(ndf_index)):
            nd_ranks[P[ndf_index[j]]] = i
        count = count + len(ndf_index)
        P_new = np.ones((len(df_index), ), dtype=np.int)
        for j in range(0, len(df_index)):
            P_new[j] = P[df_index[j]]
        P = P_new
        i = i+1
    return nd_ranks


def check_radius_rule(X, X_c, sigma, dim, nc, d_thresh=1.0):
    """ Function that implements the Radius Rule of SOP

    This function implements the radius-rule (MODIFIED) for SOP algorithm
    by checking if a new potential center is within a threshold radius of
    previously selected centers or not.

    :param X: A vector/point being considered to be selected as a center
    :type X: numpy 1D array
    :param X_c: Set of already selected centers
    :type X_c: 2D numpy array
    :param sigma: Candidate search radius
    :type sigma: Float
    :param dim: number of decision variables in X
    :type dim: int
    :param nc: number of centers (i.e., rows) in X_c
    :type nc: int
    :param d_thresh: An adaptive multiplier that allows points close to each
        other to be selected as center points, simultaneously
    :type d_thresh: float
    :return: A flag with value = 0 if X is within a distance sigma*d_thresh
        of any point in X_c and 1 otherwise
    :rtype: Bool
    """

    flag = 1
    for i in range(nc):
        if X_c[i, dim+4] == 0:
            sigma_new = sigma
        else:
            sigma_new = np.power(1/2, X_c[i, dim+4])*sigma
        d = scp.distance.euclidean(X, X_c[i, 0:dim])
        if d < sigma_new*d_thresh/np.sqrt(len(X)):
            flag = 0
            break
    return flag
