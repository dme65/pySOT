#!/usr/bin/env python
import numpy as np
import time

class Ackley:
    #  Details: http://www.cs.unm.edu/~neal.holts/dga/benchmarkFunction/ackley.html
    #  Global optimum: f(0,0,...,0)=0
    def __init__(self, dim=10):
        self.xlow = -15 * np.ones(dim)
        self.xup = 20 * np.ones(dim)
        self.dim = dim
        self.info = str(dim)+"-dimensional Ackley function \n" +\
                             "Global optimum: f(0,0,...,0) = 0"
        self.integer = []
        self.continuous = np.arange(0, dim)

    def objfunction(self, x):
        if len(x) != self.dim:
            raise ValueError('Dimension mismatch')
        n = float(len(x))
        return -20.0 * np.exp(-0.2*np.sqrt(sum(x**2)/n)) - np.exp(sum(np.cos(2.0*np.pi*x))/n)
