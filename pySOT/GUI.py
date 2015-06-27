#!/usr/bin/python
"""
..module:: GUI.py
  :synopsis: A Graphical User Interface for pySOT
..moduleauthor:: David Eriksson <dme65@cornell.edu>
"""

import sys
from PySide import QtGui, QtCore
import ntpath

ntpath.basename("a/b/c")
import imp
import os
import pySOT
import numpy as np
import importlib
from poap.controller import *
from poap.strategy import *
#import logging
import time

# Get module-level logger
logger = logging.getLogger(__name__)

# =========================== Timing ==============================

class timerThread(QtCore.QThread):
    timeElapsed = QtCore.Signal(int)

    def __init__(self, parent=None):
        super(timerThread, self).__init__(parent)
        self.timeStart = None

    def start(self, timeStart):
        self.timeStart = timeStart

        return super(timerThread, self).start()

    def run(self):
        while self.parent().isRunning():
            self.timeElapsed.emit(time.time() - self.timeStart)
            time.sleep(1)


class myThread(QtCore.QThread):
    timeElapsed = QtCore.Signal(int)
    run_timer = False

    def __init__(self, parent=None):
        super(myThread, self).__init__(parent)

        self.timerThread = timerThread(self)
        self.timerThread.timeElapsed.connect(self.timeElapsed.emit)

    def run(self):
        self.timerThread.start(time.time())

        while self.run_timer:
            time.sleep(1)

# ============================= Helpers =============================

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def path_head(path):
    head, tail = ntpath.split(path)
    return head

def get_object(module_name, object_name):
    """return method object or class object given their names"""
    module = importlib.import_module(module_name)
    obj = getattr(module, object_name)
    return obj

# ========== Manage communication with the strategy and the GUI ============

class Manager(InputStrategy):
    def __init__(self, controller):
        strategy = controller.strategy
        InputStrategy.__init__(self, controller, strategy)
        self.controller.strategy = self
        self.GUI = None
        self.numeval = 0
        self.killed = 0

    def notify(self, msg):
        pass

    def on_complete(self, rec):
        self.numeval += 1
        self.GUI.update(rec.params[0], rec.value, self.numeval, self.killed)

    def on_kill(self, rec):
        self.killed += 1

    def on_terminate(self, rec):
        self.GUI.printMessage("Optimization finished\n", "green")

    def run(self, GUI, feasible_merit):
        self.GUI = GUI
        result = self.controller.run(merit=feasible_merit)
        if result is None:
            self.GUI.printMessage("No result", "red")
        else:
            self.GUI.printMessage("Best value: " + str(result.value) + "\n", "blue")
            self.GUI.printMessage('Best solution: {0}\n'.format(
                np.array_str(result.params[0], max_line_width=50, precision=5, suppress_small=True)), "blue")

# ================================= GUI ===================================

class myGUI(QtGui.QWidget):

    def __init__(self):
        # Constructor
        super(myGUI, self).__init__()

        self.numeval = 0
        self.numfail = 0
        self.objfun = None
        self.external = False

        # Run params
        self.fbest = np.inf
        self.feasible = True
        self.xbest = None

        self.data = None
        self.nthreads = 4
        self.maxeval = 1000
        self.nsample = 4

        self.exp_des = None
        self.con_hand = None
        self.rs = None

        # Input check
        self.datainp = False
        self.threadinp = True
        self.evalinp = True
        self.siminp = True
        self.inevinp = False
        self.search = None

        # Controller
        self.controller = None
        self.manager = None
        self.myThread = myThread(self)
        self.myThread.timeElapsed.connect(self.on_myThread_timeElapsed)

        # Title
        self.titlelbl = QtGui.QLabel("Surrogate Optimization Toolbox (pySOT)", self)
        self.titlelbl.move(200, 10)

        """ File dialog """
        self.inputDlgBtn = QtGui.QPushButton("Optimization problem", self)
        self.importBtn = QtGui.QPushButton("Import", self)
        self.connect(self.inputDlgBtn, QtCore.SIGNAL("clicked()"), self.openInputDialog)
        self.connect(self.importBtn, QtCore.SIGNAL("clicked()"), self.importAction)
        self.inputline = QtGui.QLineEdit("", self)
        self.inputDlgBtn.move(0, 25)
        self.importBtn.move(510, 25)
        self.inputline.setFixedWidth(340)
        self.inputline.move(170, 30)
        self.inputDlgBtn.show()
        self.importBtn.show()

        """ Info lines """
        self.info1 = QtGui.QLabel(self)
        self.info1.move(400, 65)
        self.info2 = QtGui.QLabel(self)
        self.info2.move(400, 85)
        self.info3 = QtGui.QLabel(self)
        self.info3.move(400, 105)
        self.info4 = QtGui.QLabel(self)
        self.info4.move(400, 125)
        self.info5 = QtGui.QLabel(self)
        self.info5.move(400, 145)

        """ Input lines """

        # Number of threads
        self.threadlbl = QtGui.QLabel("Number of threads", self)
        self.threaderr = QtGui.QLabel(self)
        self.threadline = QtGui.QLineEdit("4", self)
        self.threadline.setFixedWidth(60)
        self.threadline.move(170, 60)
        self.threadlbl.move(5, 65)
        self.threaderr.move(240, 65)
        self.threadline.textChanged[str].connect(self.threadChange)
        self.threadline.show()

        # Number of evaluations
        self.evallbl = QtGui.QLabel("Maximum evaluations", self)
        self.evalerr = QtGui.QLabel(self)
        self.evalline = QtGui.QLineEdit("500", self)
        self.evalline.setFixedWidth(60)
        self.evalline.move(170, 90)
        self.evallbl.move(5, 95)
        self.evalerr.move(240, 95)
        self.evalline.textChanged[str].connect(self.evalChange)
        self.evalline.show()

        # Simultaneous evaluations
        self.simlbl = QtGui.QLabel("Simultaneous evaluations", self)
        self.simerr = QtGui.QLabel(self)
        self.simline = QtGui.QLineEdit("4", self)
        self.simline.setFixedWidth(60)
        self.simline.move(170, 120)
        self.simlbl.move(5, 125)
        self.simerr.move(240, 125)
        self.simline.textChanged[str].connect(self.simChange)
        self.simline.show()

        """ Drop-down lists """

        # Asynchronous
        self.synchlbl = QtGui.QLabel("Asynchronous", self)
        self.synchlist = QtGui.QComboBox(self)
        self.synchlist.addItem("Yes")
        self.synchlist.addItem("No")
        self.synchlist.move(170, 150)
        self.synchlbl.move(5, 155)
        self.synchlist.show()
        self.synchlist.setCurrentIndex(1)
        self.synchlist.setEnabled(False)

        # Experimental design
        self.explbl = QtGui.QLabel("Experimental design", self)
        self.explist = QtGui.QComboBox(self)
        self.explist.addItem("LatinHypercube")
        self.explist.addItem("SymmetricLatinHypercube")
        self.explist.move(170, 180)
        self.explbl.move(5, 185)
        self.explist.setCurrentIndex(1)
        self.explist.show()

        # Initial evaluations
        self.inevlbl = QtGui.QLabel("Initial evaluations", self)
        self.ineverr = QtGui.QLabel(self)
        self.inevline = QtGui.QLineEdit("", self)
        self.inevline.setFixedWidth(60)
        self.inevline.move(170, 210)
        self.inevlbl.move(5, 215)
        self.ineverr.move(240, 215)
        self.inevline.textChanged[str].connect(self.inevChange)
        self.inevline.show()

        # Search strategy
        self.searchlbl = QtGui.QLabel("Search Strategy", self)
        self.searchlist = QtGui.QComboBox(self)
        self.searchlist.addItem("CandidateDyCORS")
        self.searchlist.addItem("CandidateSRBF")
        self.searchlist.addItem("CandidateUniform")
        self.searchlist.move(170, 240)
        self.searchlbl.move(5, 245)
        self.searchlist.show()

        # Response surface
        self.rslbl = QtGui.QLabel("Response Surface", self)
        self.rslist = QtGui.QComboBox(self)
        self.rslist.addItem("Cubic Radial Basis function")
        self.rslist.addItem("Linear Radial Basis function")
        self.rslist.addItem("Thin-Plate Radial Basis function")
        self.rslist.addItem("Kriging")
        self.rslist.addItem("MARS")
        self.rslist.activated[str].connect(self.rsActivated)
        self.rslist.move(170, 270)
        self.rslbl.move(5, 275)
        self.rslist.show()

        # Tail for RBF
        self.taillbl = QtGui.QLabel("Tail function", self)
        self.taillist = QtGui.QComboBox(self)
        self.taillist.addItem("LinearTail")
        self.taillist.addItem("ConstantTail")
        self.taillist.move(170, 300)
        self.taillbl.move(5, 305)
        self.taillist.show()

        # Capping
        self.rsclbl = QtGui.QLabel("Capped Surface", self)
        self.rsclist = QtGui.QComboBox(self)
        self.rsclist.addItem("Yes")
        self.rsclist.addItem("No")
        self.rsclist.move(170, 330)
        self.rsclbl.move(5, 335)
        self.rsclist.setCurrentIndex(1)
        self.rsclist.show()

        # Controller
        self.controllerlbl = QtGui.QLabel("Controller", self)
        self.controllerlist = QtGui.QComboBox(self)
        self.controllerlist.addItem("ThreadController")
        self.controllerlist.addItem("SerialController")
        self.controllerlist.move(170, 360)
        self.controllerlbl.move(5, 365)
        self.controllerlist.activated[str].connect(self.controllerActivated)
        self.controllerlist.show()

        # Strategy
        self.stratlbl = QtGui.QLabel("Strategy", self)
        self.stratlist = QtGui.QComboBox(self)
        self.stratlist.addItem("SyncStrategyNoConstraints")
        self.stratlist.addItem("SyncStrategyPenalty")
        self.stratlist.move(170, 390)
        self.stratlbl.move(5, 395)
        self.stratlist.show()

        # Optimization button
        self.optimizebtn = QtGui.QPushButton('Optimize', self)
        self.optimizebtn.setStyleSheet("background-color: silver")
        self.optimizebtn.clicked.connect(self.optimizeActivated)
        self.optimizebtn.move(10, 420)
        self.optimizebtn.resize(80, 50)

        # Stop button
        self.stopbtn = QtGui.QPushButton('Stop', self)
        self.stopbtn.setStyleSheet("background-color: red")
        self.stopbtn.clicked.connect(self.stopActivated)
        self.stopbtn.move(100, 420)
        self.stopbtn.resize(50, 50)
        self.stopbtn.setEnabled(False)

        """ Run information """

        # Number of Evaluations
        temp = QtGui.QLabel("Completed Evals: ", self)
        temp.move(5, 480)
        self.nevallbl = QtGui.QLabel("", self)
        self.nevallbl.move(110, 480)

        # Number of crashed evaluations
        temp = QtGui.QLabel("Failed Evals: ", self)
        temp.move(5, 500)
        self.faillbl = QtGui.QLabel("", self)
        self.faillbl.move(110, 500)

        # Best value found
        temp = QtGui.QLabel("Best value: ", self)
        temp.move(5, 520)
        self.bestlbl = QtGui.QLabel("", self)
        self.bestlbl.move(110, 520)

        # Best solution feasible
        temp = QtGui.QLabel("Feasible: ", self)
        temp.move(5, 540)
        self.feaslbl = QtGui.QLabel("", self)
        self.feaslbl.move(110, 540)

        # Time
        temp = QtGui.QLabel("Time elapsed: ", self)
        temp.move(5, 560)
        self.timelbl = QtGui.QLabel("Not running", self)
        self.timelbl.move(110, 560)

        """ Log text area """

        self.log = QtGui.QTextEdit("", self)
        self.log.setReadOnly(True)
        self.log.setLineWrapMode(QtGui.QTextEdit.NoWrap)
        font = self.log.font()
        font.setFamily("Courier")
        font.setPointSize(10)
        self.log.move(180, 420)
        self.log.resize(400, 160)
        self.log.show()
        self.printMessage("Log file initiated\n")
        self.printMessage("Please import your optimization problem\n")

    def printMessage(self, text, color="blue"):
        self.log.moveCursor(QtGui.QTextCursor.End)
        self.log.setTextColor(color)
        self.log.insertPlainText(text)
        sb = self.log.verticalScrollBar()
        sb.setValue(sb.maximum())

    """ Triggers """

    def rsActivated(self, text):
        if (text == "Kriging") or (text == "MARS"):
            self.rsclist.setCurrentIndex(1)
            self.rsclist.setEnabled(False)
            self.taillist.setCurrentIndex(0)
            self.taillist.setEnabled(False)
        else:
            self.rsclist.setEnabled(True)
            self.taillist.setEnabled(True)

    def controllerActivated(self, text):
        if text == "SerialController":
            self.threadline.setText("1")
            self.threadline.setDisabled(True)
            self.simline.setText("1")
            self.simline.setDisabled(True)
        else:
            self.threadline.setDisabled(False)
            self.simline.setDisabled(False)

    def stopActivated(self):
        self.printMessage("Optimization aborted\n", "red")
        self.manager.terminate()

    def threadChange(self, text):
        if text.isdigit() and int(text) > 0:
            self.threaderr.setText("")
            self.threaderr.adjustSize()
            self.threadinp = True
        else:
            self.threaderr.setText("Invalid input!")
            self.threaderr.adjustSize()
            self.threadinp = False
        self.simChange(self.simline.text())

    def evalChange(self, text):
        if text.isdigit() and int(text) > 0:
            self.evalerr.setText("")
            self.evalerr.adjustSize()
            self.evalinp = True
        else:
            self.evalerr.setText("Invalid input!")
            self.evalerr.adjustSize()
            self.evalinp = False

    def simChange(self, text):
        if not text.isdigit():
            self.simerr.setText("Invalid input!")
            self.simerr.adjustSize()
            self.siminp = False
        elif (not self.threadline.text().isdigit()) or int(text) > int(self.threadline.text()):
            self.simerr.setText("Must be less than\nnumber of threads!")
            self.simerr.adjustSize()
            self.siminp = False
        else:
            self.simerr.setText("")
            self.simerr.adjustSize()
            self.siminp = True

    def inevChange(self, text):
        if self.data is None:
            self.ineverr.setText("Need data object!")
            self.inevinp = False
        elif text.isdigit() and int(text) > 0:
            if self.rslist.currentText() == "Kriging" or self.rslist.currentText() == "MARS":
                self.ineverr.setText("")
                self.inevinp = True
            else:
                minpts = 2*(self.data.dim + 1)  # FIXME, for now
                if int(text) >= minpts:
                    self.ineverr.setText("")
                    self.inevinp = True
                else:
                    self.ineverr.setText("Need at least " + str(minpts)+" evals")
                    self.inevinp = False
        else:
            self.ineverr.setText("Invalid input!")
            self.inevinp = False
        self.ineverr.adjustSize()

    def openInputDialog(self):
        path, _ = QtGui.QFileDialog.getOpenFileName(self, "Open File", os.getcwd())
        self.inputline.setText(path)
        self.inputline.adjustSize()

    def importAction(self):
        try:
            sys.path.append(os.path.dirname(self.inputline.text()))
            mod = imp.load_source(os.path.splitext(path_leaf(self.inputline.text()))[0],
                                  self.inputline.text())  # Load module
            class_ = getattr(mod, os.path.splitext(path_leaf(self.inputline.text()))[0])
            self.data = class_()
            pySOT.validate(self.data)
            # Check if the objective function is external
            if not hasattr(self.data, "objfunction"):
                self.external = True
                self.objfun = getattr(mod, "objfunction")
            else:
                self.external = False
                self.objfun = None
            self.printMessage("Import successful\n", "green")
            self.datainp = True
            self.inevChange(self.inevline.text())
            # Print info
            self.info1.setText(str("Problem name: " + os.path.splitext(path_leaf(self.inputline.text()))[0]))
            self.info1.adjustSize()
            self.info2.setText(str(self.data.dim) + " dimensions")
            self.info2.adjustSize()
            self.info3.setText(str(len(self.data.integer)) + " integer variables")
            self.info3.adjustSize()
            self.info4.setText("Non-box constraints: " + str(hasattr(self.data, 'eval_ineq_constraints')))
            self.info4.adjustSize()
            if self.external:
                self.info5.setText("External objective function")
            else:
                self.info5.setText("Python objective function")
            self.info5.adjustSize()

        except Exception, err:
            self.printMessage("Import failed: " + err.message + "\n", "red")
            self.data = None
            self.datainp = False
            self.external = False
            self.objfun = None
            # Reset info
            self.info1.setText("")
            self.info1.adjustSize()
            self.info2.setText("")
            self.info2.adjustSize()
            self.info3.setText("")
            self.info3.adjustSize()
            self.info4.setText("")
            self.info4.adjustSize()
            self.info5.setText("")
            self.info5.adjustSize()

    def update(self, xnew, fnew, numeval, numfail):
        # Process new information
        x = xnew.reshape((1, xnew.shape[0]))
        if hasattr(self.data, 'eval_ineq_constraints') and np.max(self.data.eval_ineq_constraints(x)) > 0.0:
            fnew = np.inf

        if fnew < self.fbest:
            self.fbest = fnew
            self.xbest = xnew
        if self.fbest < np.inf:
            self.feasible = True
        else:
            self.feasible = False

        self.numeval = numeval
        self.numfail = numfail

        # Update GUI
        self.nevallbl.setText(str(self.numeval))
        self.nevallbl.adjustSize()
        self.bestlbl.setText("{:4.4f}".format(self.fbest))
        self.bestlbl.adjustSize()
        self.feaslbl.setText(str(self.feasible))
        self.feaslbl.adjustSize()
        self.faillbl.setText(str(self.numfail))
        self.faillbl.adjustSize()

        # Force redraw
        QtGui.QApplication.processEvents()

    def paintEvent(self, event):

        qp = QtGui.QPainter()
        qp.begin(self)
        qp.drawLine(395, 60, 395, 170)
        qp.drawLine(395, 60, 595, 60)
        qp.drawLine(595, 60, 595, 170)
        qp.drawLine(395, 170, 595, 170)
        qp.drawLine(2, 475, 2, 585)
        qp.drawLine(2, 475, 178, 475)
        qp.drawLine(178, 475, 178, 585)
        qp.drawLine(2, 585, 178, 585)
        qp.end()

    @QtCore.Slot()
    def on_timer_activated(self):
        self.myThread.start()

    @QtCore.Slot(int)
    def on_myThread_timeElapsed(self, seconds):
        self.timelbl.setText("{0} seconds".format(seconds))
        self.timelbl.adjustSize()

        # Force redraw
        QtGui.QApplication.processEvents()

# ==============================================================================

    def optimizeActivated(self):

         # Are we ready for this?
        self.numeval = 0
        self.numfail = 0
        self.xbest = None
        self.fbest = np.inf
        self.feasible = np.NaN

        # Reset parameters from last solution
        self.nevallbl.setText("")
        self.nevallbl.adjustSize()
        self.bestlbl.setText("")
        self.bestlbl.adjustSize()
        self.feaslbl.setText("")
        self.feaslbl.adjustSize()
        self.faillbl.setText("")
        self.faillbl.adjustSize()

        # Timer
        self.timelbl.setText("Not running")
        self.timelbl.adjustSize()

        if not self.datainp:
            self.printMessage("No Optimization problem imported\n", "red")
        elif not self.threadinp:
            self.printMessage("Incorrect number of threads\n", "red")
        elif not self.evalinp:
            self.printMessage("Incorrect number of evaluations\n", "red")
        elif not self.siminp:
            self.printMessage("Incorrect number of simultaneous evaluations\n", "red")
        elif not self.inevinp:
            self.printMessage("Incorrect number of initial evaluations\n", "red")
        else:
            self.optimizebtn.setEnabled(False)

            # Extract parameters
            self.nthreads = int(self.threadline.text())
            self.maxeval = int(self.evalline.text())
            self.nsample = int(self.simline.text())

            # Experimental design
            try:
                exp_design = get_object('pySOT', self.explist.currentText())
                self.exp_des = exp_design(dim=self.data.dim, npts=int(self.inevline.text()))
            except Exception, err:
                self.printMessage("Failed to initialize experimental design:\n"
                                  + err.message + "\n", "red")
                self.optimizebtn.setEnabled(True)
                return

            # Search strategy
            try:
                search_strategy = get_object('pySOT', self.searchlist.currentText())
                self.search = search_strategy(data=self.data)
            except Exception, err:
                self.printMessage("Failed to initialize search strategy:\n"
                                  + err.message + "\n", "red")
                self.optimizebtn.setEnabled(True)
                return

            # Response surface Fixme Should be more general (Fix later)
            try:
                phi_ = None
                dphi_ = None
                tail_ = None
                dtail_ = None
                if self.rslist.currentText() == "Kriging":
                    self.rs = pySOT.KrigingInterpolant(maxp=self.maxeval)
                elif self.rslist.currentText() == "MARS":
                    self.rs = pySOT.MARSInterpolant(maxp=self.maxeval)
                else:
                    # Kernel
                    if self.rslist.currentText() == "Linear Radial Basis function":
                        phi_ = pySOT.phi_linear
                        dphi_ = pySOT.dphi_linear
                    elif self.rslist.currentText() == "Cubic Radial Basis function":
                        phi_ = pySOT.phi_cubic
                        dphi_ = pySOT.dphi_cubic
                    elif self.rslist.currentText() == "Thin-Plate Radial Basis function":
                        phi_ = pySOT.phi_plate
                        dphi_ = pySOT.dphi_plate

                    # Tail
                    if self.taillist.currentText() == "LinearTail":
                        tail_ = pySOT.linear_tail
                        dtail_ = pySOT.dlinear_tail
                    elif self.taillist.currentText() == "ConstantTail":
                        tail_ = pySOT.const_tail
                        dtail_ = pySOT.dconst_tail

                    # Build RBF surface
                    self.rs = pySOT.RBFInterpolant(phi=phi_, P=tail_, dphi=dphi_, dP=dtail_, maxp=self.maxeval)
            except Exception, err:
                self.printMessage("Failed to initialize response surface:\n"
                                  + err.message + "\n", "red")
                self.optimizebtn.setEnabled(True)
                return

            # Capping
            if self.rsclist.currentText() == "Yes":
                try:
                    self.rs = pySOT.RSCapped(self.rs)
                except Exception, err:
                    self.printMessage("Failed to apply capping:\n"
                                      + err.message + "\n", "red")
                    self.optimizebtn.setEnabled(True)
                    return

            try:
                # Controller
                if self.controllerlist.currentText() == "SerialController":
                    self.controller = SerialController(self.data.objfunction)
                elif self.controllerlist.currentText() == "ThreadController":
                    self.controller = ThreadController()
                # Strategy
                if self.stratlist.currentText() == "SyncStrategyNoConstraints":
                    self.controller.strategy = \
                        pySOT.SyncStrategyNoConstraints(0, self.data, self.rs, self.maxeval,
                                                        self.nsample, self.exp_des, self.search)
                elif self.stratlist.currentText() == "SyncStrategyPenalty":
                    self.controller.strategy = \
                        pySOT.SyncStrategyPenalty(0, self.data, self.rs, self.maxeval,
                                                  self.nsample, self.exp_des, self.search)
                self.controller.strategy = CheckWorkerStrategy(self.controller, self.controller.strategy)
                # Threads
                for _ in range(self.nthreads):
                    if self.external:
                        self.controller.launch_worker(self.objfun(self.controller))
                    else:
                        worker = BasicWorkerThread(self.controller, self.data.objfunction)
                        self.controller.launch_worker(worker)
            except Exception, err:
                self.printMessage("Failed to initiate controller/strategy:\n"
                                  + err.message + "\n", "red")
                self.optimizebtn.setEnabled(True)
                return

            # Optimize
            try:
                self.stopbtn.setEnabled(True)
                self.printMessage("Optimization initialized\n")

                def feasible_merit(record):
                    """Merit function for ordering final answers -- kill infeasible x"""
                    x = record.params[0].reshape((1, record.params[0].shape[0]))
                    if hasattr(self.data, 'eval_ineq_constraints') and np.max(self.data.eval_ineq_constraints(x)) > 0.0:
                        return np.inf
                    return record.value

                self.numeval = 0
                self.xbest = None
                self.fbest = np.inf

                self.manager = Manager(self.controller)

                self.on_timer_activated()
                self.myThread.run_timer = True

                self.manager.run(self, feasible_merit)
            except Exception, err:
                self.printMessage("Optimization failed:\n"
                                  + err.message + "\n", "red")

            self.myThread.run_timer = False
            time.sleep(1)
            self.printMessage("Runtime: " + self.timelbl.text() + "\n", "blue")
            self.timelbl.setText("Not running")
            self.timelbl.adjustSize()
            self.manager = None
            self.optimizebtn.setEnabled(True)
            self.stopbtn.setEnabled(False)

            # Force redraw
            QtGui.QApplication.processEvents()


def GUI():
    app = QtGui.QApplication(sys.argv)
    ex = myGUI()
    qr = ex.frameGeometry()
    cp = QtGui.QDesktopWidget().availableGeometry().center()
    qr.moveCenter(cp)
    ex.move(qr.topLeft())
    ex.setFixedSize(600, 590)
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    GUI()
