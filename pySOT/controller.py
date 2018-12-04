"""
.. module:: controller
   :synopsis: pySOT controllers

.. moduleauthor:: David Eriksson <dme65@cornell.edu>,

:Module: controller
:Author: David Eriksson <dme65@cornell.edu>,

"""

import dill
import copy
import os.path


class CheckpointController(object):
    """Checkpoint controller

    Controller that uses dill to take snapshots of the strategy each time
    an evaluation is completed, killed, or the run is terminated. We assume
    that the strategy can be pickled, or this won't work. We currently do not
    respect potential termination callbacks and failed evaluation callbacks.
    The strategy needs to implement a resume method that is called when a run
    is resumed. The strategy object can assume that all pending evaluations
    have been killed and that their respective callbacks won't be executed

    :param controller: POAP controller
    :type controller: Controller
    :param fname: Filename for checkpoint file (file cannot exist for new run)
    :type fname: string

    :ivar controller: POAP controller
    :ivar fname: Filename for snapshot
    """

    def __init__(self, controller, fname="checkpoint.pysot"):
        controller.add_feval_callback(self._add_on_update)
        controller.add_feval_callback(self.on_new_feval)
        controller.add_term_callback(self.on_terminate)
        self.controller = controller
        self.fname = fname

    def _add_on_update(self, record):
        """Internal handler -- add on_update callback to all new fevals.

        :param record: Evaluation record
        :type record: EvalRecord
        """
        record.add_callback(self.on_update)

    def on_new_feval(self, record):
        """Handle new function evaluation request.

        :param record: Evaluation record
        :type record: EvalRecord
        """
        pass

    def _save(self):
        """Save the strategy by calling the save method."""
        self.controller.strategy.save(self.fname)

    def resume(self):
        """Resume an optimization run.

        :return: The record corresponding to the best solution
        :rtype: EvalRecord
        """
        if not os.path.isfile(self.fname):
            raise IOError("Checkpoint file does not exist")
        with open(self.fname, 'rb') as input:
            self.controller.strategy = dill.load(input)
        fevals = copy.copy(self.controller.strategy.fevals)
        self.controller.fevals = fevals
        self.controller.strategy.resume()
        return self.controller.run()

    def on_update(self, record):
        """Handle feval update.

        :param record: Evaluation record
        :type record: EvalRecord
        """
        if record.is_completed:
            self.on_complete(record)
        elif record.is_killed:
            self.on_kill(record)
        elif record.is_cancelled:
            self.on_cancel(record)

    def on_complete(self, record):
        """Handle feval completion.

        :param record: Evaluation record
        :type record: EvalRecord
        """
        self._save()

    def on_kill(self, record):
        """"Handle record killed.

        :param record: Evaluation record
        :type record: EvalRecord
        """
        self._save()

    def on_cancel(self, record):
        """"Handle record cancelled.

        :param record: Evaluation record
        :type record: EvalRecord
        """
        self._save()

    def on_terminate(self):
        """"Handle termination."""
        self._save()

    def run(self):
        """Start the optimization run.

        Make sure we do not overwrite any existing checkpointing files

        :return: The record corresponding to the best solution
        :rtype: EvalRecord
        """
        if os.path.isfile(self.fname):
            raise IOError("Checkpoint file already exists, aborting...")
        return self.controller.run()
