"""
.. module:: Test_gp.py
  :synopsis: Used for Gaussian Process Regression Tests involving expected 
  improvement. 
.. moduleauthor:: Dan Liu <dl556@cornell.edu>
"""

#---------------------------------------------------------------------------
"""
TEST RESULTS with 100 points (results may vary from test to test):

2-dim Ackley-> 13.6 whereas the optimum is 0.00
Hartman3-> -3.4 whereas the optimum is -3.8
Hartman6-> -1.5 whereas the optimum is -3.3
Sphere-> 2.09 whereas the optimum is 0
Quartic-> 0.45 whereas the optimum is 0 + noise
"""
#---------------------------------------------------------------------------


from poap.controller import SerialController, ThreadController, BasicWorkerThread
import numpy as np
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from pySOT import * 
import matplotlib.pyplot as plt

#Actual test of integration. The functions chosen are arbitrary and from the 
#tutorial.

# Decide how many evaluations we are allowed to use
maxeval = 100


#Keep note of tests done a
data1 = Ackley(dim=2)
data2 = Hartman3()
data3 = Hartman6()
data4 = Sphere(dim = 2)
data5 = Quartic()

#1-----------------------------------------
exp_des = SymmetricLatinHypercube(dim=data1.dim, npts=2*data1.dim+1)
surrogate = GPRegression(20)
#surrogate = RBFInterpolant(kernel=CubicKernel, tail=LinearTail, maxp=maxeval)

adapt_samp = MultiStartGradient(data=data1)

controller = SerialController(data1.objfunction)

# (5) Use the sychronous strategy without non-bound constraints
strategy = SyncStrategyNoConstraints(
        worker_id=0, data=data1, maxeval=maxeval, nsamples=1,
        exp_design=exp_des, response_surface=surrogate,
        sampling_method=adapt_samp)
controller.strategy = strategy

# Run the optimization strategy
result = controller.run()

# Print the final result
print(data1.info)
print('Best value found: {0}'.format(result.value))
print('Best solution found: {0}'.format(
    np.array_str(result.params[0], max_line_width=np.inf,
                precision=5, suppress_small=True)))

#2----------------
exp_des = SymmetricLatinHypercube(dim=data2.dim, npts=2*data2.dim+1)
surrogate = GPRegression(20)
#surrogate = RBFInterpolant(kernel=CubicKernel, tail=LinearTail, maxp=maxeval)

adapt_samp = MultiStartGradient(data=data2)

controller = SerialController(data2.objfunction)

# (5) Use the sychronous strategy without non-bound constraints
strategy = SyncStrategyNoConstraints(
        worker_id=0, data=data2, maxeval=maxeval, nsamples=1,
        exp_design=exp_des, response_surface=surrogate,
        sampling_method=adapt_samp)
controller.strategy = strategy

# Run the optimization strategy
result = controller.run()

# Print the final result
print(data2.info)
print('Best value found: {0}'.format(result.value))
print('Best solution found: {0}'.format(
    np.array_str(result.params[0], max_line_width=np.inf,
                precision=5, suppress_small=True)))

#3-----------------
exp_des = SymmetricLatinHypercube(dim=data3.dim, npts=2*data3.dim+1)
surrogate = GPRegression(20)
#surrogate = RBFInterpolant(kernel=CubicKernel, tail=LinearTail, maxp=maxeval)

adapt_samp = MultiStartGradient(data=data3)

controller = SerialController(data3.objfunction)

# (5) Use the sychronous strategy without non-bound constraints
strategy = SyncStrategyNoConstraints(
        worker_id=0, data=data3, maxeval=maxeval, nsamples=1,
        exp_design=exp_des, response_surface=surrogate,
        sampling_method=adapt_samp)
controller.strategy = strategy

# Run the optimization strategy
result = controller.run()

# Print the final result
print(data3.info)
print('Best value found: {0}'.format(result.value))
print('Best solution found: {0}'.format(
    np.array_str(result.params[0], max_line_width=np.inf,
                precision=5, suppress_small=True)))
#4-------------------

exp_des = SymmetricLatinHypercube(dim=data4.dim, npts=2*data4.dim+1)
surrogate = GPRegression(20)
#surrogate = RBFInterpolant(kernel=CubicKernel, tail=LinearTail, maxp=maxeval)

adapt_samp = MultiStartGradient(data=data4)

controller = SerialController(data4.objfunction)

# (5) Use the sychronous strategy without non-bound constraints
strategy = SyncStrategyNoConstraints(
        worker_id=0, data=data4, maxeval=maxeval, nsamples=1,
        exp_design=exp_des, response_surface=surrogate,
        sampling_method=adapt_samp)
controller.strategy = strategy

# Run the optimization strategy
result = controller.run()

# Print the final result
print(data4.info)
print('Best value found: {0}'.format(result.value))
print('Best solution found: {0}'.format(
    np.array_str(result.params[0], max_line_width=np.inf,
                precision=5, suppress_small=True)))
#5-------------------------

exp_des = SymmetricLatinHypercube(dim=data5.dim, npts=2*data5.dim+1)
surrogate = GPRegression(20)
#surrogate = RBFInterpolant(kernel=CubicKernel, tail=LinearTail, maxp=maxeval)

adapt_samp = MultiStartGradient(data=data5)

controller = SerialController(data5.objfunction)

# (5) Use the sychronous strategy without non-bound constraints
strategy = SyncStrategyNoConstraints(
        worker_id=0, data=data5, maxeval=maxeval, nsamples=1,
        exp_design=exp_des, response_surface=surrogate,
        sampling_method=adapt_samp)
controller.strategy = strategy

# Run the optimization strategy
result = controller.run()

# Print the final result
print(data5.info)
print('Best value found: {0}'.format(result.value))
print('Best solution found: {0}'.format(
    np.array_str(result.params[0], max_line_width=np.inf,
                precision=5, suppress_small=True)))