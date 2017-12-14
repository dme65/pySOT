"""
.. module:: gp_regression
   :synopsis: Gaussian Process regression

.. moduleauthor:: David Eriksson <dme65@cornell.edu>

:Module: gp_regression
:Author: David Eriksson <dme65@cornell.edu>

:Small additions made by Dan Liu <dl556@cornell.edu>
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from scipy.stats import norm
from scipy import optimize



class GPRegression(GaussianProcessRegressor):
    """Compute and evaluate a GP

    Gaussian Process Regression object.

    Depends on scitkit-learn==0.18.1.

    More details:
        http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html

    Note that the last four variables are included solely for the knowledge
    gradient calculation.     

    :param maxp: Initial capacity
    :type maxp: int
    :param gp: GP object (can be None)
    :type gp: GaussianProcessRegressor

    :ivar nump: Current number of points
    :ivar maxp: Initial maximum number of points (can grow)
    :ivar x: Interpolation points
    :ivar fx: Function evaluations of interpolation points
    :ivar gp: Object of type GaussianProcessRegressor
    :ivar dim: Number of dimensions
    :ivar model: MARS interpolation model
    :ivar kernelm: Contains the variance matrix of the current set of points
    :ivar linFuncts: Contains the linear functions used to calculate the
    knowledge gradient
    :ivar cvals: Contains the intersection points of the above linear
    functions
    :ivar relevantFuncts: Contains the relevant linear functions 
    that correspond to the minimum of the linear functions at each point
    :ivar lastPoint: Used for the knowledge gradient to check whether
    the required a/b/c values (see Scott, Powell and Frazier 2011)
    """

    def __init__(self, maxp=100, gp=None):
        self.nump = 0
        self.maxp = maxp
        self.x = None     # pylint: disable=invalid-name
        self.fx = None
        self.dim = None
        if gp is None:
            self.model = GaussianProcessRegressor(n_restarts_optimizer=10)
        else:
            self.model = gp
            if not isinstance(gp, GaussianProcessRegressor):
                raise TypeError("gp is not of type GaussianProcessRegressor")
        self.updated = False

        #Additional variables added 
        self.kernelm = None
        self.linFuncts = None
        self.cvals = None
        self.relevantFuncts = None
        self.lastPoint = None



    def reset(self):
        """Reset the interpolation."""

        self.nump = 0
        self.x = None
        self.fx = None
        self.updated = False

    def _alloc(self, dim):
        """Allocate storage for x, fx, rhs, and A.

        :param dim: Number of dimensions
        :type dim: int
        """

        maxp = self.maxp
        self.dim = dim
        self.x = np.zeros((maxp, dim))
        self.fx = np.zeros((maxp, 1))

    def _realloc(self, dim, extra=1):
        """Expand allocation to accommodate more points (if needed)

        :param dim: Number of dimensions
        :type dim: int
        :param extra: Number of additional points to accommodate
        :type extra: int
        """

        if self.nump == 0:
            self._alloc(dim)
        elif self.nump+extra > self.maxp:
            self.maxp = max(self.maxp*2, self.maxp+extra)
            #TODO: REMOVE REFCHECK STUFF
            self.x.resize((self.maxp, dim), refcheck = False)
            self.fx.resize((self.maxp, 1), refcheck = False)

    def get_x(self):
        """Get the list of data points

        :return: List of data points
        :rtype: numpy.array
        """

        return self.x[:self.nump, :]

    def get_fx(self):
        """Get the list of function values for the data points.

        :return: List of function values
        :rtype: numpy.array
        """

        return self.fx[:self.nump, :]

    def add_point(self, xx, fx):
        """Add a new function evaluation

        :param xx: Point to add
        :type xx: numpy.array
        :param fx: The function value of the point to add
        :type fx: numpy.array
        """

        dim = len(xx)
        self._realloc(dim)
        
        if self.nump == 0:
            self.kernelm = [[1]]
        else:
            x = np.expand_dims(xx, axis=0)
            #kernel_tt = kernel(self.get_x(),x)
            kernel_tt = np.zeros((self.get_x().shape[0],1))
            for i in range ((self.get_x().shape[0])):
                 point = self.x[i,:]
                 diffs_squared = -0.5*np.sum(np.square(np.subtract(xx,point)))
                 kernel_tt[i,0] = np.exp(diffs_squared)
       
            kernelm = np.concatenate((self.kernelm, kernel_tt), axis = 1)
            kernel_ttplusnew = np.concatenate((kernel_tt,[[1]]),axis = 0)
            self.kernelm = np.concatenate((kernelm, kernel_ttplusnew.T), axis = 0)

        self.x[self.nump, :] = xx
        self.fx[self.nump, :] = fx
        self.nump += 1
        self.updated = False
        

    def eval_old(self, x):
        """Evaluate the GP regression object at the point x. Because 
        this is not used within the optimization framework, it is given
        the label "old" for now.

        :param x: Point where to evaluate
        :type x: numpy.array
        :return: Value of the GP regression obejct at x
        :rtype: float
        """

        if self.updated is False:
            self.model.fit(self.get_x(), self.get_fx())
        self.updated = True

        x = np.expand_dims(x, axis=0)
        fx = self.model.predict(x)
        return fx

    def evals(self, x, ds=None):
        """Evaluate the GP regression object at the points x.

        :param x: Points where to evaluate,of size npts x dim
        :type x: numpy.array
        :param ds: Not used
        :type ds: None
        :return: Values of the GP regression object at x, of length npts
        :rtype: numpy.array
        """

        if self.updated is False:
            self.model.fit(self.get_x(), self.get_fx())
        self.updated = True

        fx = np.zeros((1,x.shape[0]))
        for i in range (x.shape[0]):
            fx[0,i] = self.eval(x[i,:])

        return fx.flatten()    


    def deriv_old(self, x, ds=None):
        """Evaluate the gradient of the GP regression object at a point x.
        Because it does not use expected improvement, it is classified as 
        "old"

        :param x: Point for which we want to compute the GP regression gradient
        :type x: numpy.array
        :param ds: Not used
        :type ds: None
        :return: Derivative of the GP regression object at x
        :rtype: numpy.array
        """

        # Implemented using an approximation (first differences)
        # rather than analytically

        if self.updated is False:
            self.model.fit(self.get_x(), self.get_fx())
        self.updated = True

        eps = np.sqrt(np.finfo(np.float).eps)
        grad = optimize.approx_fprime (x, self.eval, eps)
        return grad 


    def eval(self, x, ds = None):
        """Evaluate the EXPECTED IMPROVEMENT of the GP object at the point x.
        The expected improvement is calculated using the formula given in 
        Jones[1998]. 

        :param x: Points where to evaluate, of size npts x dim
        :type x: numpy.array
        :return: Values of the expected improvement of the
        GP regression object at x, of length npts
        :rtype: numpy.array
        """
        if self.updated is False:
            self.model.fit(self.get_x(), self.get_fx())
        self.updated = True
        x = np.expand_dims(x, axis=0)

        fx, fx_std = self.model.predict(x,return_std = True)

        #We assume that the minimum of the function occurs at the minimum of
        #the currently sampled values (located in self.fx)

        #Warning: sklearn sometimes returns the predicted standard deviation as 0.
        #In those cases we assume that the expected improvement is 0
        #This may or may not be changed in the future
        if (fx_std != 0):
            current_min = np.amin(self.get_fx())
            val_to_check = (current_min - fx)/fx_std
            pdf_fx= norm.pdf(val_to_check)
            cdf_fx = norm.cdf(val_to_check)

            #We use the formula from Jones 1998 for expected improvement 
            exp_imp = (current_min - fx) *cdf_fx + fx_std * pdf_fx 
        else:
            exp_imp = np.zeros((1,1))
        return exp_imp.flatten()
        
    def deriv_old2(self, x):
        """Evaluate the gradient of the EXPECTED IMPROVEMENT
        of the GP regression object at a point x. Note: The gradient is
        approximated using first differences in each dimension. 

        :param x: Point for which we want to compute the GP regression gradient
        :type x: numpy.array
        :return: Gradient of the GP regression object at x
        :rtype: numpy.array
        """
        
        if self.updated is False:
            self.model.fit(self.get_x(), self.get_fx())
        self.updated = True

        # Implemented using an approximation (first differences)
        # rather than analytically
        eps = np.sqrt(np.finfo(np.float).eps)
        grad = optimize.approx_fprime (x, self.eval, eps)
        return grad

    def deriv(self, x, ds = None):
        """Evaluate the gradient of the EXPECTED IMPROVEMENT
        of the GP regression object at a point x analytically.

        :param x: Point for which we want to compute the GP regression gradient
        :type x: numpy.array
        :return: Gradient of the GP regression object at x
        :rtype: numpy.array
        """

        #WORKS! Comparison with first differences values yields good results
        #Note: This function assumes that the kernel function has the form
        #exp(-0.5(dist(x_1, x_2)^2))
        #Note: May be slow. Will optimize later.

        if self.updated is False:
            self.model.fit(self.get_x(), self.get_fx())
        self.updated = True
        x = np.expand_dims(x, axis=0)

        fx, fx_std = self.model.predict(x,return_std = True)

        #We recalculate the SD and function estimate ...
        if (fx_std == 0):
            return np.zeros(x.size)

        current_min = np.amin(self.get_fx())
        val_to_check = (current_min - fx)/fx_std
        pdf_fx= norm.pdf(val_to_check)
        cdf_fx = norm.cdf(val_to_check)
           
        #As in, dEI/dx = dEi/df*df/dx + dEI/ds*ds/dx by the multivar chain rule

        #Recall that the kernel fctn is e^{-0.5*dists^2} where the distance
        #fctn is the standard Euclidean fctn 

        #First we calculate dEi/df, given by Jones 
        dEidf = -cdf_fx 

        #df/dx can be calculated using some linear equations.
        #We know that f = k(z^T)k(X)a + mean where z is the new element, 
        #X is the training data, k is the kernel fctn
        #and a is the solution to Ka = y where 
        #K is the kernel matrix already created

        #We can extract the alphas from the GPR function itself. Then we get
        #that the gradient is simply f = dk*/dx(alpha), which is equal to
        # k multiplied by (alpha) multiplied by the specific axis elementwise 
        #summed . 

        alpha = self.model.alpha_
        kernel = self.model.kernel_
        
        #Ktesttrain is a column vector with the number of rows corresponding
        #to the number of previously accessed points
        kernel_test_train = kernel(self.get_x(),x)
        #This gives df/dx without including the chain rule portion
        #of the derivative of the kernel
        #E.g. the kernel has the form e^f(x) so we have a np.multiply by
        #e^f(x), but we still have to multiply by df(x)/dx
        #That's what coord_diffs accounts for, since f(x) is the squared
        #euclidean multiplied by -0.5 
        #Note: Broadcasting is done here
        #Note 2: in the squared Euclidean we have (x2 - x1) where x2 is the
        #old point and x2 is the new point (the point to look at)
        chain_first_part = np.multiply(kernel_test_train, alpha) 
        coord_diffs = self.get_x() - x;
        dfdx = np.dot(chain_first_part.T, coord_diffs)

        #Next we calculate dEi/ds, given by Jones
        dEids = pdf_fx 

        #We know that var = k(z^T)k(z^T) + k(z^t)K(X)K^(-1)k(x^T)k(z) 
        #Thus we calculate ds/dv and dv/dx; ds/dv = 1/(2s)
        #dv/dx is :      
        kernel_inv = np.linalg.inv(self.kernelm)

        #This gives the derivative of dK/dx where K is the kernel of the 
        #test and training matrix    
        chain_first_part2 = np.multiply(coord_diffs, kernel_test_train)
        #We use the fact that dy^TAy for symmetric matrices is
        #2Ay, and then we apply dK/dx as above
        dvdx = 2 * np.dot (np.dot(kernel_test_train.T, kernel_inv), chain_first_part2)
        #The previous equations took the derivative of the variance, but we 
        #want the derivative of the SD, so we have one more part  
        dsdx = -dvdx /(2*fx_std)

        #We FINALLY use the chain rule and get...
        grad = dEidf * dfdx + dEids *dsdx 

        return grad.flatten()

    def eval_kn_grad(self, x, var=None, calc = True):
        """Evaluate the point with respect to the knowledge gradient as
        discussed by Powell and Frazier (2010). The variance is required
        to account for noisy calculations. Note that if the variance is 0,
        then this method defaults to expected improvement where the point
        sampled is excluded from the subtraction. Note also that the method
        given is framed in terms of a maximization, so the code here
        has been modified to account for this. 


        :param x: Points where to evaluate, of size npts x dim
        :type x: numpy.array
        :param var: The variance function
        :type var: ...?(functionType) TODO
        :return: Values of the expected improvement of the
        GP regression object at x, of length npts
        :rtype: numpy.array
        """

        #The idea is to calculate the maximum value of 
        #the difference between the minimum of the estimated value of x_new
        #given the current gaussian regression of the function
        #and the expected value of the new minimum after the gaussian regression
        #is modified to include the new point

        #This is functionally equivalent to expected improvement with
        #no variance except for the fact that we also include x_new
        #as one of the points in the minuend (whereas expected improvement
        #assumes that the minimum given the current gaussian regression function
        #is at one of the already previously sampled points)

        #Strictly speaking if we know that the variance is none it would
        #be easier simply to use the expected improvement formula
        #since this formula is a lot more calculation intensive while returning
        #the same result.

        #The first thing we do is rewrite the expected value formula in terms
        #of finding the minimum value among a set of linear functions of the form
        #a + by.
        #In this case, a is the prediction of the value of the point x_i given
        #the current gaussian regression, b is the value of the kernalized
        #product between x_i and x_new, and y is the independent variable
        #corresponding to the ACTUAL value of the function at x_new versus
        #the value of the function at x_new predicted given the current
        #gaussian prediction divided by the standard deviation of the aforementioned
        #value. 

        #Note that in this case, there are n-1 values of a and b corresponding to
        #n-1 previously sampled values. Thus this calculation becomes more
        #expensive as the number of points increases. 

        if self.updated is False:
            self.model.fit(self.get_x(), self.get_fx())
        self.updated = True
        x = np.expand_dims(x, axis=0)

        alpha = self.model.alpha_
        kernel = self.model.kernel_

        #Store the linear functions as described above (the +1 accounts
        #for the new point) (each column corresponds to one linear function)
        linFuncts = np.zeros((2, self.nump+1))

        #Here we get the "b" values
        #Var_fun is simply the kernel matrix if the data is NOT noisy     
        var_fun = kernel(x,self.get_x())
        #Include the new point in "var fun" (kernel of point with itself is 1)
        var_fun_with_new = np.append(var_fun, 1)

        #Here we get the 'a'  values
        exp_vals = self.get_fx().T 
        #Append the current prediction to the end
        fx, fx_std = self.model.predict(x,return_std = True)
        exp_vals = np.append(exp_vals, fx)


        assert self.nump+1 == exp_vals.shape[0]
        assert self.nump+1 == var_fun_with_new.shape[0]

        #Store values of a and b in lin funct
        linFuncts[0] = exp_vals
        linFuncts[1] = var_fun_with_new
        #Also stored for the use of the derivative function (see below)
        coord_diffs = self.get_x() - x;
        bgrad = np.multiply(coord_diffs, var_fun.T)
        #Include the gradient of the kernel function of x_new and itself
        #This vector is zero (the kernel remains at 1 regardless of the value
        #of x new)
        bgrad = np.concatenate((bgrad, [np.zeros(x.shape[1])]))
        linFuncts = np.concatenate((linFuncts,bgrad.T),axis = 0) 

        #Part two..rearranging linfuncts so that the slopes (b values) are arranged
        #in descending order..
        linFuncts = np.fliplr(linFuncts[:,np.argsort(linFuncts[1,:])])
        #Part three: find 'c' values that denote intersection of above functions
        #to find the expected minimum value

        #The vector A refers to the indices of lines that are relevant for
        #finding the minimum of the linear functions (by column)
        #The vector c refers to the c values that correspond to the intersection
        #points of these valid lines 

        #We also need to do a check in case the b values are the same, in which
        #case, we discard the linear function with a higher value of a, since
        #we want to find the piecewise minimum of all the linear functions
        #over the entire domain

        A = np.arange(self.nump+1)
        while True:
            to_discard = []
            c_old = None
            for i in range (len(A)-1):
                if (linFuncts[1, A[i+1]] - linFuncts[1, A[i]] == 0):
                    if (linFuncts[0, A[i+1]] > linFuncts[0, A[i]]):
                        to_discard.append(A[i+1])
                    else:
                        to_discard.append(A[i])
                else: 
                    c_calc = ( (linFuncts[0,A[i]] - linFuncts[0, A[i+1]])/
                        (linFuncts[1, A[i+1]] - linFuncts[1, A[i]]) )
                    if c_old is not None:
                        if c_calc < c_old: 
                            to_discard.append(A[i])
                    c_old = c_calc 
            #Remove unneeded lines 
            for i in range (len(to_discard)):
                #A.remove(to_discard[i]) 
                A = np.delete(A, np.where(A == to_discard[i]), axis=0)
            if len(to_discard) == 0:
                break

        #Presumably all the functions are needed now. Do one more check to
        #get the c values
        c_vals = np.zeros(len(A)-1)
        for i in range (len(A)-1):
            c_calc = ( (linFuncts[0,A[i]] - linFuncts[0, A[i+1]])/
                (linFuncts[1, A[i+1]] - linFuncts[1, A[i]]) )
            c_vals[i] = c_calc 

        #If "calc", then we actually calculate the knowledge gradient.
        #Sometimes we just want to set up the global variables
        #(e.g. we want the gradient of the knowledge gradient but
        #don't care about the knowledge gradient itself)
        if calc: 
            #Now we have values of A and c. We use these to get the expected value.
            #This method is taken from Scott, Frazier, and Powell 2011
            e_sum = 0      
            for i in range (len(A)):
                if i == 0:
                    sum_a = linFuncts[0,A[0]]*norm.cdf(c_vals[0])
                    sum_b = -linFuncts[1,A[0]]*norm.pdf(c_vals[0])
                    e_sum = e_sum + sum_a + sum_b
                elif i == len(A)-1:
                    sum_a = linFuncts[0, A[i]]*(1-norm.cdf(c_vals[i-1]))
                    sum_b = linFuncts[1, A[i]]*(norm.pdf(c_vals[i-1]))
                    e_sum = e_sum + sum_a + sum_b
                else:
                    sum_a = linFuncts[0, A[i]]*(norm.cdf(c_vals[i]) - norm.cdf(c_vals[i-1]))
                    sum_b = linFuncts[1, A[i]]*(norm.pdf(c_vals[i-1]-norm.pdf(c_vals[i])))
                
            #Then e_sum corresponds to the expected value. Now we actually have to
            #get the knowledge gradient
            current_min = np.amin(self.get_fx())
            if fx < current_min:
                current_min = fx


        #Function updates
        self.linFuncts = linFuncts
        self.cvals = c_vals
        self.relevantFuncts = A
        self.lastPoint = x

        return (current_min - e_sum).flatten() 

    def eval_kn_grad_der(self, x):
        """Evaluate the gradient (Scott, Frazier and Powell (2011)
        of the (knowledge gradient)
        using the GP regression object at a point x analytically.

        :param x: Point for which we want to compute the GP regression gradient
        :type x: numpy.array
        :return: Gradient of the GP regression object at x
        :rtype: numpy.array
        """

        #Assume self.linFuncts, self.cvals and self.relevantFuncts are
        #given properly, or set them up if not...
        for i in range (len(x)):
            if self.lastPoint.flatten()[i] != x.flatten()[i]:
                eval_kn_grad(x,calc = False)
                self.lastPoint = x
                break


        #Part 1: Calculate the gradient of the value of x_i wrt changes in
        #x_n. This is the first portion of the sum calculated in the paper
        #and is expressed by grad(u(x_i)).
        #Note that the gradient is only zero unless x_i = x_n, in which
        #case we have to calculate the derivative of the gaussian process
        #prediction wrt a change in the input. This was already done
        #in the expected improvement section above so I have copy-pasted the
        #code here...    

        if self.updated is False:
            self.model.fit(self.get_x(), self.get_fx())
        self.updated = True
        x = np.expand_dims(x, axis=0)

        alpha = self.model.alpha_
        kernel = self.model.kernel_

        #This is taken from the code on expected improvement and assumes
        #that the variance function is zero... (otherwise alpha is technically
        #incorrect in this case since alpha doesn't account for the variance
        #function)
        #Ktesttrain is a column vector with the number of rows corresponding
        #to the number of previously accessed points
        kernel_test_train = kernel(self.get_x(),x)
        chain_first_part = np.multiply(kernel_test_train, alpha) 
        coord_diffs = self.get_x() - x;
        dfdx = np.dot(chain_first_part.T, coord_diffs).T

        #Note: REMOVE THIS CODE
        # #In this matrix I place the a_i values according to the order of 
        # #the A values
        # ai_grad = np.zeros(len(x), len(self.relevantFuncts)-1)
        # for i in range(len(self.relevantFuncts)):
        #     if self.linFuncts(2, self.relevantFuncts(i)) == 0:
        #         ai_grad(:,i) = dfdx

        #Part 2: Calculate the gradient of the particular variance function
        #wrt x_n where in this case the variance is equal to the kernel matrix
        #multiplication between the two points. This function also doesn't
        #account for the variance.
        #Note: each row accounts for the gradient of x_i with x_n where i
        #refers to the row
        #This gradient has already been calculated in the eval_kn_gradient
        #and is included in self.linFuncts

        #Part 3: Calculate the gradient of the of the c_values (found in the
        #third part of the sum given in the 2011 paper)

        #Note that for the particular paper we have from 4.14 that
        #we need to take the gradients of the a_i values and the b_i values
        #The a_i values correspond the u_n(x_i), so the gradient of the
        #a_i value is 0 unless x_i = x_n as in part 1
        #The b_i values correspond to the gradient of the kernelized product
        #calculated in part 2 above

        #Note that I skip c0 = -infty and c(len(A)) = infty (since according
        #to the paper they equal 0)
        ci_grad = np.zeros((len(x.flatten()), len(self.relevantFuncts)-1))
        for i in range (len(self.relevantFuncts)-1):
            fr = self.relevantFuncts[i]
            sd = self.relevantFuncts[i+1]
            var_dif = self.linFuncts[1,sd] - self.linFuncts[1, fr]
            if (self.linFuncts[2,fr] == 0):
                sum_1 = dfdx * var_dif
            elif (self.linFuncts[2, sd] == 0):
                sum_1 = -dfdx * var_dif
            else:    
                sum_1 = 0

            sum_1 = sum_1.flatten()
            sum_2 = (self.linFuncts[2:,sd]  \
                - self.linFuncts[2:,fr])  \
                * (self.linFuncts[0, fr] - self.linFuncts[0, sd])
            #sum_2 = np.expand_dims(sum_2, axis=0).T
            denom = var_dif**2
            ci_grad[:,i] = (sum_1 - sum_2)/denom 


        #Part 4: Calculate the actual sum given in the paper. 
        total_sum = 0
        for i in range (len(self.relevantFuncts)):
            #There are three parts here as well
            #The first corresponds to part 1 described above
            #The second corresponds to part 2 described above
            #The third corresponds to the gradient of the 'c' values
            #described in part 3 above
            #We also need special checks for i == 0 and i == len(A) since
            #there the c values involved include c = -inf and c = +inf
            #We need a special check when A[i] corresponds to a value of
            #a_i where a_i = u_n(x_n) since the gradient for a_i in calculating
            #c_i is only non-zero at that point
            #We do this by checking the gradient of the variance, which 
            #is zero for that value (since the variance is always equal to 1)
            #Note that indexing of cvals is done wrt to i-1 and i; this is 
            #due to the fact that the cvals array does not include c0 = -infty
            # and cn = infty
            actFunct = self.relevantFuncts[i]
            if i == 0:
                #P1 
                if self.linFuncts[2, actFunct] == 0:
                    sump1 = norm.cdf(self.cvals[0])*dfdx
                else:
                    sump1 = np.zeros(2)
                #P2 
                sump2 = -self.linFuncts[2:,actFunct] * norm.pdf(self.cvals[0])
                #P3
                sump3 = (self.linFuncts[0, actFunct] + self.linFuncts[1, actFunct] \
                    *self.cvals[i])*norm.pdf(self.cvals[i]) * ci_grad[:,0]
            elif i == len(self.relevantFuncts)-1:
                #P1
                if (self.linFuncts[2, actFunct] == 0):
                    sump1 = (1 - norm.cdf(self.cvals[i-1]))*dfdx
                else: 
                    sump1 = np.zeros(2)
                #P2
                sump2 = self.linFuncts[2:,actFunct] * (norm.pdf(self.cvals[i-1]))
                #P3 
                sump3 = - (self.linFuncts[0, actFunct] + self.linFuncts[1, actFunct] \
                    *self.cvals[i-1])*norm.pdf(self.cvals[i-1]) * ci_grad[:,i-1]
            else:
                #P1
                if self.linFuncts[2, actFunct] == 0:
                    sump1 = (norm.cdf(self.cvals[i])-norm.cdf(self.cvals[i-1])) \
                        *dfdx
                else:
                    sump1 = [0, 0]
                #P2
                sump2 = self.linFuncts[2:,actFunct]*(norm.pdf(self.cvals[i-1]) \
                    -norm.pdf(self.cvals[i]))
                #P3 
                sump3 = (self.linFuncts[0, actFunct] + self.linFuncts[1, actFunct] \
                    *self.cvals[i])*norm.pdf(self.cvals[i]) * ci_grad[:,i] \
                    - (self.linFuncts[0, actFunct] + self.linFuncts[1, actFunct] \
                    *self.cvals[i-1])*norm.pdf(self.cvals[i-1]) * ci_grad[:,i-1]
             
            sump1 = sump1.flatten()
            sump2 = sump2.flatten()
            sump3 = sump3.flatten()
            total_sum = total_sum + sump1 + sump2 + sump3

        return total_sum
# ====================================================================


def _main():

    def test_f(x):
        fx = x[0]^2 + x[1]^2
        return fx    

    # # Set up GP model: expected improvement 
    fhat = GPRegression(20)    

    #As of now, the expected improvement calculation is used
    #to get the derivatives etc. 
    fhat.add_point([3,1], test_f([3,1]))
    x = np.array([5.1,5.1])
    #Analytic gradient 
    #print(fhat.eval(x))
    print("Analytic gradient")
    print (fhat.deriv(x));
    #Finite differences gradient 
    print ("Finite differences gradient.")
    print (fhat.deriv_old2(x)); 


    #Testing with more points 
    fhat.add_point([4,2], test_f([4,2]))
    fhat.add_point([5,3], test_f([5,3]))
    print("Analytic gradient")
    print (fhat.deriv(x));
    print ("Finite differences gradient.")
    print (fhat.deriv_old2(x));   

    #Test evals
    x = np.zeros((3,2))
    #print(x)
    for i in range(3):
        x[i,0] = i
        x[i,1] = i+1

    print("Evals check")
    print(fhat.evals(x))    

    #Test with knowledge gradient 
    fhat2 = GPRegression(20)


    fhat2.add_point([3,1], test_f([3,1]))
    fhat.add_point([4,2], test_f([4,2]))
    x = np.array([5.1,5.1])
    #Analytic gradient 
    print("Analytic knowledge gradient")
    print (fhat2.eval_kn_grad(x));
    #Finite differences gradient 
    print("Analytic knowledge gradient gradient")
    print (fhat2.eval_kn_grad_der(x));   

if __name__ == "__main__":
    _main()
