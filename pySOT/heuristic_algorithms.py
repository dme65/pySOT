"""
.. module:: heurisitc_algorithms
   :synopsis: A mixed-integer GA implementation for bound constrained problems
.. moduleauthor:: David Eriksson <dme65@cornell.edu>
"""
__author__ = 'davideriksson'

import numpy as np
from experimental_design import LatinHypercube


class GeneticAlgorithm:
    def __init__(self, function, dim, xlow, xup, intvar=[], popsize=100, ngen=100, start="LHD"):
        self.nVariables = dim
        self.nIndividuals = popsize + (popsize % 2)  # Make sure this is even
        self.lowerBoundary = np.array(xlow)
        self.upperBoundary = np.array(xup)
        self.integerVariables = np.array(intvar)
        self.start = start
        self.sigma = 0.2
        self.pMutation = 1.0/dim
        self.tournamentSize = 5
        self.pCross = 0.9
        self.numberOfGenerations = ngen
        self.function = function

    def optimize(self):
        #  Initialize population
        if isinstance(self.start, np.ndarray):
            assert self.start.shape[0] == self.nIndividuals and self.start.shape[1] == self.nVariables
            assert all(np.min(self.start, axis=0) >= self.lowerBoundary) and \
                all(np.max(self.start, axis=0) <= self.upperBoundary)
            population = self.start
        elif self.start == "LHD":
            exp_des = LatinHypercube(self.nVariables, self.nIndividuals)
            population = self.lowerBoundary + exp_des.generate_points() * \
                (self.upperBoundary - self.lowerBoundary)
        elif self.start == "Random":
            population = self.lowerBoundary + np.random.rand(self.nIndividuals, self.nVariables) *\
                (self.upperBoundary - self.lowerBoundary)
        else:
            raise AttributeError("Unknown argument for initial population")

        newpopulation = []
        #  Round positions
        if len(self.integerVariables) > 0:
            newpopulation = np.copy(population)
            population[:, self.integerVariables] = np.round(population[:, self.integerVariables])
            for i in self.integerVariables:
                ind = np.where(population[:, i] < self.lowerBoundary[i])
                population[ind, i] += 1
                ind = np.where(population[:, i] > self.upperBoundary[i])
                population[ind, i] -= 1

        #  Evaluate all individuals
        functionValues = self.function(population)
        if len(functionValues.shape) == 2:
            functionValues = np.squeeze(np.asarray(functionValues))

        # Save the best individual
        ind = np.argmin(functionValues)
        bestIndividual = np.copy(population[ind, :])
        bestValue = functionValues[ind]

        if len(self.integerVariables) > 0:
            population = newpopulation

        # Main loop
        for ngen in range(self.numberOfGenerations):
            # Do tournament selection to select the parents
            competitors = np.random.randint(0, self.nIndividuals, (self.nIndividuals, self.tournamentSize))
            ind = np.argmin(functionValues[competitors], axis=1)
            winnerIndices = np.zeros(self.nIndividuals, dtype=int)
            for i in range(self.tournamentSize):  # This loop is short
                winnerIndices[np.where(ind == i)] = competitors[np.where(ind == i), i]

            parent1 = population[winnerIndices[0:self.nIndividuals/2], :]
            parent2 = population[winnerIndices[self.nIndividuals/2:self.nIndividuals], :]

            # Averaging Crossover
            cross = np.where(np.random.rand(self.nIndividuals/2) < self.pCross)[0]
            nn = len(cross)  # Number of crossovers
            alpha = np.random.rand(nn, 1)

            # Create the new chromosomes
            parent1New = np.multiply(alpha, parent1[cross, :]) + np.multiply(1-alpha, parent2[cross, :])
            parent2New = np.multiply(alpha, parent2[cross, :]) + np.multiply(1-alpha, parent1[cross, :])
            parent1[cross, :] = parent1New
            parent2[cross, :] = parent2New
            population = np.concatenate((parent1, parent2))

            # Apply mutation
            scalefactors = self.sigma * (self.upperBoundary - self.lowerBoundary)  # Account for dimensions ranges
            perturbation = np.random.randn(self.nIndividuals, self.nVariables)  # Generate perturbations
            perturbation = np.multiply(perturbation, scalefactors)  # Scale accordingly
            perturbation = np.multiply(perturbation, (np.random.rand(self.nIndividuals,  # Determine where to perturb
                                                                     self.nVariables) < self.pMutation))
            population += perturbation  # Add perturbation
            population = np.maximum(np.reshape(self.lowerBoundary, (1, self.nVariables)), population)
            population = np.minimum(np.reshape(self.upperBoundary, (1, self.nVariables)), population)

            # Round chromosomes
            newpopulation = []
            #  Round positions
            if len(self.integerVariables) > 0:
                newpopulation = np.copy(population)
                population[:, self.integerVariables] = np.round(population[:, self.integerVariables])
                for i in self.integerVariables:
                    ind = np.where(population[:, i] < self.lowerBoundary[i])
                    population[ind, i] += 1
                    ind = np.where(population[:, i] > self.upperBoundary[i])
                    population[ind, i] -= 1

            # Keep the best individual
            population[0, :] = bestIndividual

            #  Evaluate all individuals
            functionValues = self.function(population)
            if len(functionValues.shape) == 2:
                functionValues = np.squeeze(np.asarray(functionValues))

            # Save the best individual
            ind = np.argmin(functionValues)
            bestIndividual = np.copy(population[ind, :])
            bestValue = functionValues[ind]

            if len(self.integerVariables) > 0:
                population = newpopulation

        return bestIndividual, bestValue

def main():

    dim = 30

    # Vectorized Ackley function in dim dimensions
    def objfunction(x):
        return -20.0*np.exp(-0.2*np.sqrt(np.sum(x**2, axis=1)/dim)) - \
            np.exp(np.sum(np.cos(2.0*np.pi*x), axis=1)/dim)

    ga = GeneticAlgorithm(objfunction, dim, -15*np.ones(dim), 20*np.ones(dim), popsize=100, ngen=100, start="SLHD")
    xBest, fBest = ga.optimize()

    # Print the best solution found
    print("\nBest function value: %f" % fBest)
    print("Best solution: " % xBest)
    np.set_printoptions(suppress=True)
    print(xBest)

if __name__ == "__main__":
    main()

