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
    """Widget containing a plot and toolbar.

    Attributes:
        figure: Matplotlib Figure object, containing all plot elements.
        plotFigure: Matplotlib FigureCanvas object containing plot figure and
            navigation toolbar.
    """

    def __init__(self):
        """Initializes plot widget"""
        super().__init__()
        self.setupPlot()

    def setupPlot(self):
        """Create canvas containing plot and navigation toolbar."""
        plotLayout = QVBoxLayout()
        self.figure = Figure(tight_layout={"pad": 2.0})
        self.plotFigure = FigureCanvas(self.figure)
        toolbar = NavigationToolbar(self.plotFigure, self)
        plotLayout.addWidget(self.plotFigure, 1)
        plotLayout.addWidget(toolbar)
        self.setLayout(plotLayout)


class PlotAndTable(QTabWidget):
    """Widget containing plot on tab 1, and a table on tab 2.

    Attributes:
        plotWidget: PlotWidget object, contains plot and toolbar.
        figure: Matplotlib Figure object, contains all plot elements.
        tableWidget: QTableView object, contains data in table format.
    """

    def __init__(self, plotTabLabel, tableTabLabel):
        """Initializes PlotAndTable class.

        Args:
            plotTabLabel: Text label (string) for plot tab.
            tableTabLabel: Text label (string) for table tab.
        """
        super().__init__()
        self._setupPlotTab(plotTabLabel)
        self._setupTableTab(tableTabLabel)

    def _setupPlotTab(self, plotTabLabel):
        """Creates plot widget and figure.

        Args:
            plotTabLabel: Text label (string) for plot tab.
        """
        self.plotWidget = PlotWidget()
        self.figure = self.plotWidget.figure
        self.addTab(self.plotWidget, plotTabLabel)

    def _setupTableTab(self, tableTabLabel):
        """Creates table widget.

        Args:
            tableTabLabel: Text label (string) for table tab.
        """
        self.tableWidget = QTableView()
        self.addTab(self.tableWidget, tableTabLabel)


class ComputeWidget(QWidget):
    """Handles running estimation, showing progress on separate window.

    Attributes:
        computeTask: TaskThread object, performs the estimation calculations on
            a separate thread.
        results: pyqtSignal, emits dict containing model objects (with
            estimation results as properties) as values, indexed by name of
            model/metric combination.
        _progressBar: QProgressBar object, indicates the progress of the
            estimation calculations.
        _numCombinations: Total number of estimation calculations to perform.
            Equal to the number of models selected times the number of metric
            combinations.
        _label: QLabel object, text displayed on the progress window showing
            which model/metric combination is currently being calculated, and
            how many combinations have been calculated out of the total.
        _modelCount: The number of combinations that have completed the
            estimation calculations.
    """
    results = pyqtSignal(dict)

    def __init__(self, modelsToRun, metricNames, data, config, parent=None):
        """Initializes ComputeWidget class.

        Args:
            modelsToRun: List of Model objects used for estimation calculation.
            metricNames: List of metric names as strings used for estimation
                calculation.
            data: Pandas dataframe containing imported data.
            config: ConfigParser object containing information about which model
                functions are implemented. Passed to Model.
            parent:
        """
        super(ComputeWidget, self).__init__(parent)
        layout = QVBoxLayout(self)

        # set fixed window size (width, height)
        self.setFixedSize(350, 200)

        self._progressBar = QProgressBar(self)
        self._numCombinations = len(modelsToRun) * len(metricNames)
        self._progressBar.setMaximum(self._numCombinations)
        self._label = QLabel()
        self._label.setText("Computing results...\nModels completed: {0}".format(0))
        self._modelCount = 0

        layout.addWidget(self._label)
        layout.addWidget(self._progressBar)
        layout.setAlignment(Qt.AlignVCenter)
        self.setWindowTitle("Processing...")

        self.computeTask = TaskThread(modelsToRun, metricNames, data, config)
        self.computeTask.nextCalculation.connect(self._showCurrentCalculation)
        self.computeTask.modelFinished.connect(self._modelFinished)
        self.computeTask.taskFinished.connect(self._onFinished)
        self.computeTask.start()

        self.show()

    def _showCurrentCalculation(self, calcName):
        """Shows name of model combination currently being calculated """
        self._label.setText("Computing {0}...\nModels completed: {1} of {2}".format(calcName, self._modelCount, self._numCombinations))

    def _modelFinished(self):
        """Increments count of completed calculations, updates progress bar."""
        self._modelCount += 1
        self._progressBar.setValue(self._modelCount)

    def _onFinished(self, result):
        """Emits all estimation results when completed."""
        self.results.emit(result)
        self.close()


class TaskThread(QThread):
    """Runs estimation calculations on separate thread.

    Attributes:
        abort: Boolean indicating if the app has been closed. If True, the
            thread should stop running.
        modelFinished: pyqtSignal, emits when current model/metric calculation
            is completed. Tells thread to begin calculation on next
            combination.
        nextCalculation: pyqtSignal, emits string containing the model/metric
            combination name currently being calculated. Displayed on progress
            window.
        taskFinished: pyqtSignal, emits dict containing model objects (with
            estimation results as properties) as values, indexed by name of
            model/metric combination.
        _modelsToRun: List of Model objects used for estimation calculation.
        _metricNames: List of metric names as strings used for estimation
            calculation.
        _data: Pandas dataframe containing imported data.
        _config: ConfigParser object containing information about which model
                functions are implemented. Passed to Model.
    """
    taskFinished = pyqtSignal(dict)
    modelFinished = pyqtSignal()
    nextCalculation = pyqtSignal(str)

    def __init__(self, modelsToRun, metricNames, data, config):
        """Initializes TaskThread class.

        Args:
            modelsToRun: List of Model objects used for estimation calculation.
            metricNames: List of metric names as strings used for estimation
                calculation.
            data: Pandas dataframe containing imported data.
            config: ConfigParser object containing information about which model
                functions are implemented. Passed to Model.
        """
        super().__init__()
        self.abort = False  # True when app closed, so thread stops running
        self._modelsToRun = modelsToRun
        self._metricNames = metricNames
        self._data = data
        self._config = config

    def run(self):
        """Performs estimation for models/metrics.

        Called when thread is started.
        """
        # window says that symbolic equations are being calculated
        self.nextCalculation.emit("symbolic equations")
        while(not SymbolicThread.complete):
            # check if application has been closed
            if self.abort:
                return  # get out of run method
            # wait until symbolic equations are calculated, do nothing until then
            pass
        result = {}
        for model in self._modelsToRun:
            for metricCombination in self._metricNames:
                # check if application has been closed
                if self.abort:
                    return  # get out of run method
                metricNames = ", ".join(metricCombination)
                if (metricCombination == ["No covariates"]):
                    metricCombination = []
                m = model(data=self._data.getData(), metricNames=metricCombination, config=self._config)

                # this is the name used in tab 2 and tab 4 side menus
                # use shortened name
                runName = m.shortName + " (" + metricNames + ")"  # "Model (Metric1, Metric2, ...)"
                self.nextCalculation.emit(runName)
                m.runEstimation()
                result[runName] = m
                self.modelFinished.emit()
        self.taskFinished.emit(result)


class SymbolicThread(QThread):
    """Runs symbolic computation for newly imported data on its own thread.

    Attributes:
        abort: Boolean indicating if the app has been closed. If True, the
            thread should stop running.
        taskFinished: pyqtSignal, emits dict containing model objects (with
            estimation results as properties) as values, indexed by name of
            model/metric combination.
        _modelsToRun: List of Model objects used for estimation calculation.
        _metricNames: List of metric names as strings used for estimation
            calculation.
        _data: Pandas dataframe containing imported data.
        _modelFinished: pyqtSignal, emits when current model/metric calculation
            is completed. Tells thread to begin calculation on next
            combination.
        _nextCalculation: pyqtSignal, emits string containing the model/metric
            combination name currently being calculated. Displayed on progress
            window.
        _config: ConfigParser object containing information about which model
                functions are implemented. Passed to Model.
    """

    symbolicSignal = pyqtSignal()
    complete = False    # if symbolic calculations are complete = True

    def __init__(self, modelList, data, config):
        """Initializes TaskThread class.

        Args:
            modelList: List of Model objects used for symbolic calculations.
            data: Pandas dataframe containing imported data.
            config: ConfigParser object containing information about which model
                functions are implemented. Passed to Model.
        """
        super().__init__()
        self.abort = False  # True when app closed, so thread stops running
        self._modelList = modelList
        self._data = data
        self._config = config

    def run(self):
        """Performs symbolic calculations for models.

        Called when thread is started.
        """
        # log.info(f"modelList = {models.modelList}")
        # SymbolicThread.complete = False # set flag to False when starting calculations
        log.info("RUNNING SYMBOLIC THREAD")
        for model in self._modelList.values():
            # check if application has been closed
            if self.abort:
                return  # get out of run method

            # if not model.dLLF:
            if self._config[model.__name__]['dLLF'].lower() != 'yes':
                # need to initialize models so they have the imported data
                instantiatedModel = model(data=self._data.getData(),
                metricNames=self._data.metricNames, config=self._config)
            

                # # only run symbolic calculation if no LLF implemented by user
                # if not instantiatedModel.LLFspecified:
                #     pass
                # # or if LLF not created for all covariates
                # elif instantiatedModel.LLFspecified:
                #     if len(instantiatedModel.LLF_array) <= instantiatedModel.maxCovariates or None in instantiatedModel.LLF_array:


                model.lambdaFunctionAll = instantiatedModel.symAll()    # saved as class variable for each model
                log.info("Lambda function created for %s model", model.name)
        SymbolicThread.complete = True  # calculations complete, set flag to True
        self.symbolicSignal.emit()
