Surrogate optimization
======================

Surrogate optimization algorithms generally consist of four components:

1. **Optimization problem:** All of the available information about the
   optimization problem, e.g., dimensionality, variable types, objective
   function, etc.
2. **Surrogate model:** Approximates the underlying objective function.
   Common choices are RBFs, Kriging, MARS, etc.
3. **Experimental design:** Generates an initial set of points for building
   the initial surrogate model
4. **Adaptive sampling:** Method for choosing evaluations after the
   experimental design has been evaluated.

The surrogate model (or response surfaces) is used to approximate an underlying
function that has been evaluated for a set of points. During the optimization
phase information from the surrogate model is used in order to guide the search
for improved solutions, which has the advantage of not needing as many function
evaluations to find a good solution. Most surrogate model algorithms consist of
the same steps as shown in the algorithm below.

The general framework for a Surrogate Optimization algorithm is the following:

**Inputs:** Optimization problem, Experimental design, Adaptive sampling method,
Surrogate model, Stopping criterion, Restart criterion

.. code-block:: console
   :linenos:

   Generate an initial experimental design
   Evaluate the points in the experimental design
   Build a Surrogate model from the data
   Repeat until stopping criterion met
      Restart criterion met
      Reset the Surrogate model and the Sample point strategy
      go to 1
   Use the adaptive sampling method to generate new point(s) to evaluate
   Evaluate the point(s) generated using all computational resources
   Update the Surrogate model

**Outputs:** Best solution and its corresponding function value

Typically used stopping criteria are a maximum number of allowed function
evaluations (used in this toolbox), a maximum allowed CPU time, or a maximum
number of failed iterative improvement trials.
