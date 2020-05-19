# PyQt5 imports for UI elements
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTabWidget, QTableView, \
                            QProgressBar, QLabel
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtCore import Qt

# Matplotlib imports for graphs/plots
from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5
from matplotlib.backends.backend_qt5agg import FigureCanvas, \
                                    NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import logging as log

class PlotWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setupPlot()

    def setupPlot(self):
        plotLayout = QVBoxLayout()
        self.figure = Figure(tight_layout={"pad": 2.0})
        self.plotFigure = FigureCanvas(self.figure)
        toolbar = NavigationToolbar(self.plotFigure, self)
        plotLayout.addWidget(self.plotFigure, 1)
        plotLayout.addWidget(toolbar)
        self.setLayout(plotLayout)

class PlotAndTable(QTabWidget):
    """
    A widget containing a plot on tab 1, and a table on tab 2.
    Inherited from QTabWidget.
    """
    def __init__(self, plotTabLabel, tableTabLabel):
        """
        Initializes PlotAndTable class.

        Args:
            plotTabLabel (string): text label for plot tab
            tableTabLabel (string): text label for table tab
        """

        super().__init__()
        self.setupPlotTab(plotTabLabel)
        self.setupTableTab(tableTabLabel)

    def setupPlotTab(self, plotTabLabel):
        # Creating plot widget
        self.plotWidget = PlotWidget()
        self.figure = self.plotWidget.figure
        self.addTab(self.plotWidget, plotTabLabel)

    def setupTableTab(self, tableTabLabel):
        self.tableWidget = QTableView()
        self.addTab(self.tableWidget, tableTabLabel)

class ComputeWidget(QWidget):
    results = pyqtSignal(dict)

    # predict points?
    def __init__(self, modelsToRun, metricNames, data, parent=None):
        super(ComputeWidget, self).__init__(parent)
        layout = QVBoxLayout(self)
        
        # set fixed window size (width, height)
        self.setFixedSize(350, 200)

        self.progressBar = QProgressBar(self)
        self.numCombinations = len(modelsToRun) * len(metricNames)
        self.progressBar.setMaximum(self.numCombinations)
        self.label = QLabel()
        self.label.setText("Computing results...\nModels completed: {0}".format(0))
        self.modelCount = 0

        layout.addWidget(self.label)
        layout.addWidget(self.progressBar)
        layout.setAlignment(Qt.AlignVCenter)
        self.setWindowTitle("Processing...")

        self.computeTask = TaskThread(modelsToRun, metricNames, data)
        self.computeTask.nextCalculation.connect(self.showCurrentCalculation)
        self.computeTask.modelFinished.connect(self.modelFinished)
        self.computeTask.taskFinished.connect(self.onFinished)
        self.computeTask.start()

        self.show()

    def showCurrentCalculation(self, calcName):
        """ Shows name of model combination currently being calculated """
        self.label.setText("Computing {0}...\nModels completed: {1} of {2}".format(calcName, self.modelCount, self.numCombinations))

    def modelFinished(self):
        self.modelCount += 1
        self.progressBar.setValue(self.modelCount)
        # self.label.setText("Computing results...\nModels completed: {0}".format(self.modelCount))

    def onFinished(self, result):
        self.results.emit(result)
        self.close()


class TaskThread(QThread):
    modelFinished = pyqtSignal()
    taskFinished = pyqtSignal(dict)
    nextCalculation = pyqtSignal(str)

    # predict points?
    def __init__(self, modelsToRun, metricNames, data):
        super().__init__()
        self.abort = False  # True when app closed, so thread stops running
        self.modelsToRun = modelsToRun
        self.metricNames = metricNames
        self.data = data

    def run(self):
        self.nextCalculation.emit("symbolic equations") # window says that symbolic equations are being calculated
        while(not SymbolicThread.complete):
            # check if application has been closed
            if self.abort:
                return  # get out of run method
            pass    # wait until symbolic equations are calculated, do nothing until then
        result = {}
        for model in self.modelsToRun:
            for metricCombination in self.metricNames:
                # check if application has been closed
                if self.abort:
                    return  # get out of run method
                metricNames = ", ".join(metricCombination)
                if (metricCombination == ["No covariates"]):
                    metricCombination = []
                m = model(data=self.data.getData(), metricNames=metricCombination)
                runName = m.name + " - (" + metricNames + ")"  # "Model (Metric1, Metric2, ...)"
                self.nextCalculation.emit(runName)
                m.runEstimation()
                result[runName] = m
                self.modelFinished.emit()
        self.taskFinished.emit(result)

class SymbolicThread(QThread):
    """
    Runs the symbolic computation for newly imported data on its own thread
    """

    symbolicSignal = pyqtSignal()
    complete = False    # if symbolic calculations are complete = True

    def __init__(self, modelList, data):
        super().__init__()
        self.abort = False  # True when app closed, so thread stops running
        self.modelList = modelList
        self.data = data

    def run(self):
        # log.info(f"modelList = {models.modelList}")
        # SymbolicThread.complete = False # set flag to False when starting calculations
        log.info("RUNNING SYMBOLIC THREAD")
        for model in self.modelList.values():
            # check if application has been closed
            if self.abort:
                return  # get out of run method
            # need to initialize models so they have the imported data
            instantiatedModel = model(data=self.data.getData(), metricNames=self.data.metricNames)
            model.lambdaFunctionAll = instantiatedModel.symAll()    # saved as class variable for each model
            log.info("Lambda function created for %s model", model.name)
        SymbolicThread.complete = True  # calculations complete, set flag to True
        self.symbolicSignal.emit()