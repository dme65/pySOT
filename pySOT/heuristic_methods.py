"""
.. module:: heuristic_methods
   :synopsis: Heuristic optimization methods

.. moduleauthor:: David Eriksson <dme65@cornell.edu>

:Module: heuristic_methods
:Author: David Eriksson <dme65@cornell.edu>

"""

from pySOT.experimental_design import LatinHypercube, SymmetricLatinHypercube
import numpy as np


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


if __name__ == "__main__":
    dim = 30

    # Vectorized Ackley function in dim dimensions
    def obj_function(x):
        return -20.0*np.exp(-0.2*np.sqrt(np.sum(x**2, axis=1)/dim)) - \
            np.exp(np.sum(np.cos(2.0*np.pi*x), axis=1)/dim) + 20 + np.exp(1)

    ga = GeneticAlgorithm(obj_function, dim, -15*np.ones(dim), 20*np.ones(dim),
                          popsize=100, ngen=100, start="SLHD")
    x_best, f_best = ga.optimize()

    # Print the best solution found
    print("\nBest function value: {0}".format(f_best))
    print("Best solution: {0}".format(x_best))

    #  Add constraint of unit-1 norm and supply projection method

    def projection(x):
        return x / np.linalg.norm(x)

    ga = GeneticAlgorithm(obj_function, dim, -1*np.ones(dim), 1*np.ones(dim), popsize=100,
                          ngen=100, start="SLHD", projfun=projection)
    x_best, f_best = ga.optimize()

    # Print the best solution found
    print("\nBest function value: {0}".format(f_best))
    print("Best solution: {0}".format(x_best))
    print("norm(x_best) = {0}".format(np.linalg.norm(x_best)))
