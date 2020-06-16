import logging
import math

import numpy as np
import scipy.spatial as scp

from ..auxiliary_problems import candidate_dycors
from ..utils import POSITIVE_INFINITY, check_radius_rule, nd_sorting
from .surrogate_strategy import SurrogateBaseStrategy

# Get module-level logger
logger = logging.getLogger(__name__)


class _SopRecord:
    """A custom record that stores memory attributes of a SOP-related record

    A multi-attribute record that stores the evaluation point and corresponding
    attributes including objective function value, failure count, elapsed tabu
    count, non-domination rank and search radius. Failure count, tabu count,
    rank and sigma are updated after a new function evaluation is completed.

    :param x: Decision variable
    :type x: numpy array
    :param fx: objective function value
    :type fx: float
    :param sigma: Candidate search radius
    :type sigma: float
    """

    def __init__(self, x, fx, sigma):
        self.x = x
        self.fx = fx
        self.rank = POSITIVE_INFINITY  # To-Do: Update ranks in future
        self._nfail = 0  # Count of failures(int)
        self._ntabu = 0  # Elapsed tabu tenure
        self._sigma = sigma

    @property
    def sigma(self):
        """Get value of radius / sigma"""
        return self._sigma

    @property
    def nfail(self):
        """Get failure count"""
        return self._nfail

    @property
    def ntabu(self):
        """Get elapsed tabu tenure count"""
        return self._ntabu

    def reduce_sigma(self):
        """Reduce sigma / search radius"""
        self._sigma = self._sigma / 2.0

    def increment_failure_count(self):
        """Increase failure count"""
        self._nfail += 1

    def make_tabu(self, sigma):
        """Make this point tabu"""
        self._nfail = 0
        self._ntabu = 1
        self._sigma = sigma

    def increment_tabu_tenure(self):
        """Increment the elapsed tabu tenure"""
        self._ntabu += 1

    def reset(self, sigma):
        """Reset memory attributes"""
        self._nfail = 0
        self._ntabu = 0
        self._sigma = sigma


class _SopCenter:
    """ A custom reference record that stores information for a SOP center

    A multi-attribute reference record that stores the decision vector value
    of a SOP center, and correspondingly, its location in the list of evaluated
    SOP Records, the new point it generates and location of new point in list
    of evaluated records.

    :param xc: decision vector value of center
    :type xc: numpy array
    :param index: index location in list of evaluated points
    :type index: int
    """

    def __init__(self, xc, index):
        self.xc = xc
        self.index = index
        self._new_point = None  # New point proposed for eval around xc
        self._new_index = None  # Location of new point in list of evals

    @property
    def new_point(self):
        """Get new proposed point"""
        return self._new_point

    @new_point.setter
    def new_point(self, value):
        """Set new point, raise error if array length is diff from self.xc"""
        if len(value) != len(self.xc):
            raise ValueError("Dimension mismatch between center and new point")
        else:
            self._new_point = value

    @property
    def new_index(self):
        """Get location / index of new point"""
        return self._new_index

    @new_index.setter
    def new_index(self, value):
        """Set location of new point, raise error if not integer"""
        if not isinstance(value, int):
            raise ValueError("Index location is not an integer")
        else:
            self._new_index = value


class SOPStrategy(SurrogateBaseStrategy):
    """Surrogate Optimization with Pareto Selection Strategy

    This is an implementation of the SOP strategy by Krityakierne et. al:

    Tipaluck Krityakierne, Taimoor Akhtar and Christine A. Shoemaker.
    SOP: parallel surrogate global optimization with Pareto \
        center selection for computationally expensive \
        single objective problems.
    Journal of Global Optimization, 66(3), 2016.

    The core idea of SOP is to maintain a ranked archive of all previously
    evaluated points, as per non-dominated sorting between two objectives,
    i.e.,
        i) Objective function value (minimize) and
        ii)Minimum distance from
    other evaluated points(maximize). A sub-archive of center points is
    subsequently maintained via selection from the ranked evalauted points.
    The number of  points in the sub-archive of centers should be equal to
    the number of parallel threads (or greater than). Candidate points are
    generated around each 'center point' via the DYCORS sampling strategy,
    i.e., an N(0, sigma^2) distributed perturbation of a subset of decision
    variables. A separate value of sigma is maintained for each center
    point, where sigma is decreased if no progress is registered in the
    bi-objective objective value and distance criterion trade-off. One point
    is selected for expensive evaluation from each set of candidate points,
    based on the surrogate approximation only. Hence the merit function is
    s(x), where s(x) is the surrogate prediction.

    This strategy has two additional arguments than the base class:

    ncenters: Specify no. of centers (should be greater than no. of threads)
              Default = 4
    num_cand: Number of candidate to use when generating new evaluations
              Default = 100 * dim

    :param max_evals: Evaluation budget
    :type max_evals: int
    :param opt_prob: Optimization problem object
    :type opt_prob: OptimizationProblem
    :param exp_design: Experimental design object
    :type exp_design: ExperimentalDesign
    :param surrogate: Surrogate object
    :type surrogate: Surrogate
    :param ncenters: Number of center points
    :type ncenters:  int
    :param asynchronous: Whether or not to use asynchrony (True/False)
    :type asynchronous: bool
    :param batch_size: Size of batch (Make sure batch_size<=ncenters for sync)
    :type batch_size: int
    :param extra_points: Extra points to add to the experimental design
    :type extra_points: numpy.array of size n x dim
    :param extra_vals: Values for extra_points (np.nan/np.inf if unknown)
    :type extra_vals: numpy.array of size n x 1
    :param reset_surrogate: Whether or not to reset surrogate model
    :type reset_surrogate: bool
    :param use_restarts: Whether or not to restart after convergence
    :type use_restarts: bool
    :param num_cand: Number of candidate points, default = 100*dim
    :type num_cand: int
    """

    def __init__(
        self,
        max_evals,
        opt_prob,
        exp_design,
        surrogate,
        ncenters=4,
        asynchronous=True,
        batch_size=None,
        extra_points=None,
        extra_vals=None,
        use_restarts=True,
        num_cand=None,
    ):

        self.dtol = 1e-3 * math.sqrt(opt_prob.dim)

        if num_cand is None:
            num_cand = 100 * opt_prob.dim
        self.num_cand = num_cand

        self.sampling_radius = 0.2
        self.record_queue = []  # Completed records that haven't been processed
        self.num_exp = exp_design.num_pts  # We need this later
        self.ncenters = ncenters
        self.evals = []  # List of all eval points stored as _SOPRecord
        self.centers = []  # List of current center points as _SOPCenter
        self.F_ranked = None  # Evaluated points stored as numpy array
        self.d_thresh = 1.0

        super().__init__(
            max_evals=max_evals,
            opt_prob=opt_prob,
            exp_design=exp_design,
            surrogate=surrogate,
            asynchronous=asynchronous,
            batch_size=batch_size,
            extra_points=extra_points,
            extra_vals=extra_vals,
            use_restarts=use_restarts,
        )

    def check_input(self):
        """Check inputs."""
        super().check_input()
        if not isinstance(self.ncenters, int) and self.ncenters > 3:
            raise ValueError("ncenters should be an integer greater than 3")
        if not self.asynchronous:
            if not self.ncenters >= self.batch_size:
                raise ValueError("Batch size should be less than or equal" " to ncenters")

    def on_initial_completed(self, record):
        """Handle completed evaluation in initial phase"""
        super().on_initial_completed(record)

        if record.ev_id >= self.ev_last:
            srec = _SopRecord(np.copy(record.params[0]), record.value, self.sampling_radius)
            self.evals.append(srec)

    def on_adapt_completed(self, record):
        """Handle completed evaluation in phase 2."""
        super().on_adapt_completed(record)

        if record.ev_id >= self.ev_last:
            self.record_queue.append(record)

            # Initiate a new SOP Record for new completed evaluation
            center_index = None
            srec = _SopRecord(np.copy(record.params[0]), record.value, self.sampling_radius)
            self.evals.append(srec)

            ncenters = len(self.centers)
            for i in range(ncenters):  # Update location of new point in center
                if np.array_equal(np.copy(record.params[0]), self.centers[i].new_point):
                    self.centers[i].new_index = self.num_evals - 1
                    center_index = i
                    break

            if self.asynchronous:  # Process immediately
                self.adjust_memory(center_index)
            elif (not self.batch_queue) and self.pending_evals == 0:  # Batch
                self.adjust_memory()

    def generate_evals(self, num_pts):
        """Generate the next adaptive sample points."""

        # Update the list of center points
        if self.F_ranked is None:  # If this is the start of adaptive phase
            self.update_ranks()
        self.update_center_list()

        # Compute dycors perturbation probability
        num_evals = len(self.X) + len(self.Xpend) - self.num_exp + 1.0
        min_prob = np.min([1.0, 1.0 / self.opt_prob.dim])
        budget = self.max_evals - self.num_exp
        prob_perturb = min([20.0 / self.opt_prob.dim, 1.0]) * (1.0 - (np.log(num_evals) / np.log(budget)))
        prob_perturb = max(prob_perturb, min_prob)

        # Perturb each center to propose one new eval per center
        new_points = np.zeros((num_pts, self.opt_prob.dim))
        weights = [1.0]
        for i in range(num_pts):
            # Deduce index of next available center
            center_index = 0
            for center in self.centers:
                if center.new_point is None:
                    break
                center_index += 1
            # Select new point by candidate search around center
            X_c = self.centers[center_index].xc
            sampling_radius = self.evals[self.centers[center_index].index].sigma
            new_points[i, :] = candidate_dycors(
                num_pts=1,
                opt_prob=self.opt_prob,
                surrogate=self.surrogate,
                X=self._X,
                fX=self._fX,
                weights=weights,
                sampling_radius=sampling_radius,
                num_cand=self.num_cand,
                Xpend=self.Xpend,
                prob_perturb=prob_perturb,
                xbest=X_c,
            )

            self.centers[center_index].new_point = new_points[i, :]

        # submit the new points
        for i in range(num_pts):
            self.batch_queue.append(np.copy(np.ravel(new_points[i, :])))

    def adjust_memory(self, index=None):
        """Update the memory attributes of evaluated points

        For each evaluated point (stored as _SOPRecord instance) update
        i) Failure count, ii) Tabu status and iii) Sampling radius.

        """

        if index is None:  # Batch mode - update memory for all centers
            indices = range(self.ncenters)
        else:  # asynchronous mode
            indices = [index]

        # Re-evaluate bi-objective ranks after adding new point(s)
        # NOTE: Re-evaluation needed because minimum distance is updated
        # for all points
        nevals = self.num_evals
        self.update_ranks()

        # Step 2 -- Adjust memory attributes of center point associated
        # with new eval by checking if we succeeded at improving the
        # distance-objective tradeoff
        for i in indices:
            cp = self.centers[i]
            center_index = cp.index
            check = 0
            new_index = cp.new_index
            rank = self.F_ranked[new_index, self.opt_prob.dim + 3]
            if rank == 1:  # new point is in the non-dominated front
                check = 1  # success
            if check == 0:  # If no success increase failure count
                self.evals[center_index].increment_failure_count()
                self.evals[center_index].reduce_sigma()

        # Step 3 --- Update tabu list, i.e., i) include a center in
        # tabu list if its failure count is more than n_fail and ii)
        # remove a center from tabu list if it has been in the tabu
        # list for more than n_tenure iterations
        for i in range(nevals):  # check if pts are to be removed from tabu
            if self.evals[i].ntabu > 0:
                if self.evals[i].ntabu < 5:  # NOTE: Tabu tenure is 5
                    self.evals[i].increment_tabu_tenure()
                else:
                    self.evals[i].reset(self.sampling_radius)

        for i in indices:  # add a point to Tabu list if failures > fail_thresh
            cp = self.centers[i]
            index = cp.index
            if self.evals[index].nfail > 3:  # NOTE: max failure count is 4
                self.evals[index].make_tabu(self.sampling_radius)

        self.update_F()  # make sure that memory is updated in ranked F

    def update_F(self):
        """Update F_ranked numpy array"""
        nevals = self.num_evals
        F = np.zeros((nevals, self.opt_prob.dim + 5))
        F[:, 0 : self.opt_prob.dim] = [
            (val.x - self.opt_prob.lb) / (self.opt_prob.ub - self.opt_prob.lb) for val in self.evals
        ]
        F[:, self.opt_prob.dim] = [val.fx for val in self.evals]
        F[:, self.opt_prob.dim + 2] = [val.ntabu for val in self.evals]
        F[:, self.opt_prob.dim + 3] = [val.rank for val in self.evals]
        F[:, self.opt_prob.dim + 4] = [val.nfail for val in self.evals]
        self.F_ranked = np.copy(F)

    def update_ranks(self):
        """Updated ND ranks of evaluated points

        Non-dominated ranks of evaluated points are updated after
        new points have been evaluated.

        """
        nevals = self.num_evals
        self.update_F()
        F = np.copy(self.F_ranked)
        dists = scp.distance.cdist(F[:, 0 : self.opt_prob.dim], F[:, 0 : self.opt_prob.dim])
        for i in range(nevals):
            a = dists[i, :]
            F[i, self.opt_prob.dim + 1] = -1.0 * np.min(a[np.nonzero(a)])

        nmax = 100  # Maximum number of points that may be selected as centers
        ranks = nd_sorting(F[:, self.opt_prob.dim : self.opt_prob.dim + 2].transpose(), nmax)  # Perform ND Sorting
        F[:, self.opt_prob.dim + 3] = ranks.transpose()
        self.F_ranked = np.copy(F)

    def update_center_list(self):
        """This method for Updating the list of centers"

        This function updates the list of center points after new points
        have been evaluated. In a synchronous setting where batch_size
        = number of centers, the set of old centers is simply replaced
        by new centers. Otherwise, only the center point that was just
        processed, is replaced by a new center. Centers are selected from
        all evaluated points, after they are sorted according to i) ND rank
        and ii) Objective function value (Tabu points are pushed to the end
        in selection order).

        """
        nevals = self.num_evals
        F = np.copy(self.F_ranked)
        self.d_thresh = 1.0 - float(nevals - self.num_exp) / float(self.max_evals - self.num_exp)

        # Step 1 - Remove center points around which new points have been
        # proposed and evaluated
        if len(self.centers) > 0:
            finished_centers = []
            for i in range(len(self.centers)):
                if self.centers[i].new_index is not None:
                    finished_centers.append(i)
                else:  # If a center is being processed, tag it as tabu
                    F[self.centers[i].index, self.opt_prob.dim + 2] = 2
            for index in reversed(finished_centers):
                self.centers.pop(index)

        # Step 2 - Sort all evaluated points according to
        # i) tabu status, ii) rank, iii) obj value
        ind = np.lexsort((F[:, self.opt_prob.dim], F[:, self.opt_prob.dim + 3], F[:, self.opt_prob.dim + 2]))
        min_index = np.argmin(F[:, self.opt_prob.dim])
        if min_index == ind[0] or F[min_index, self.opt_prob.dim + 2] == 2:
            ind_new = np.copy(ind)
        else:  # Put xbest at the top of sorted points (ind_new),
            # regardless of tabu status unless it is being processed
            ind_new = np.copy(ind)
            ind_new[0] = min_index
            check = 0
            i = 1
            while check == 0:
                ind_new[i] = ind[i - 1]
                if ind[i] == min_index:
                    check = 1
                i = i + 1

        # Append new points to list of centers until length of centers is not
        # equal to number of centers
        num_pending_centers = len(self.centers)
        center_count = self.ncenters - num_pending_centers
        if center_count > 0:
            center_index = -1 * np.ones((center_count,), dtype=np.int)
            center_index[0] = ind_new[0]
            check = 1
            i = 1
            while check < center_count:
                if i < nevals:
                    flag = check_radius_rule(
                        F[ind_new[i], 0 : self.opt_prob.dim],
                        F[center_index, :],
                        self.sampling_radius,
                        self.opt_prob.dim,
                        check,
                        d_thresh=self.d_thresh,
                    )  # Radius Rule
                    if flag == 1:
                        center_index[check] = ind_new[i]
                        check = check + 1
                    i = i + 1
                else:
                    check_prev = check
                    while check < center_count:
                        center_index[check] = center_index[np.remainder(check, check_prev)]
                        check = check + 1
            # Initialize the center point list
            for index in center_index:
                crec = _SopCenter(self.evals[index].x, index)
                self.centers.append(crec)
