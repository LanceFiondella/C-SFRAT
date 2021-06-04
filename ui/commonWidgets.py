# To check platform
import sys

# PyQt5 imports for UI elements
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTabWidget, QTableView, \
                            QProgressBar, QLabel
#Temp Imports
##########################
from PyQt5.QtWidgets import QTableWidget, QAbstractScrollArea, QHeaderView
##########################

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtCore import Qt

from core.graphing import PlotWidget
from core.prediction import prediction_psse
from core.goodnessOfFit import PSSE


class PlotAndTable(QTabWidget):
    """Widget containing plot on tab 1, and a table on tab 2.

    Attributes:
        plotWidget: PlotWidget object, contains plot and toolbar.
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
        # self.figure = self.plotWidget.figure
        self.addTab(self.plotWidget, plotTabLabel)

    def _setupTableTab(self, tableTabLabel):
        """Creates table widget.

        Args:
            tableTabLabel: Text label (string) for table tab.
        """
        self.tableWidget = QTableView()

        self.tableWidget.setEditTriggers(QTableWidget.NoEditTriggers)     # make cells unable to be edited
        self.tableWidget.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
                                                                    # column width fit to contents

        self.addTab(self.tableWidget, tableTabLabel)

#Temporary Test
###########################################################

class TableTabs(QTabWidget):

    def __init__(self, table1Label, table2Label):
        super().__init__()
        self._setupTable1Tab(table1Label)
        self._setupTable2Tab(table2Label)

    def _setupTable1Tab(self, table1Label):
        self.budgetTab = QTableWidget()
        self.budgetTab.setEditTriggers(QTableWidget.NoEditTriggers)  # make cells unable to be edited
        self.budgetTab.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
                                                            # column width fit to contents
        self.budgetTab.setRowCount(1)
        self.budgetTab.setColumnCount(3)
        self.budgetTab.setHorizontalHeaderLabels(["Model Name", "Covariates", "Est. Defects"])

        header = self.budgetTab.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)

        # only want to change style sheet for Windows
        # on other platforms with dark modes, creates light font on light background
        if sys.platform == "win32":
            # windows
            # provides bottom border for header
            stylesheet = "::section{Background-color:rgb(250,250,250);}"
            header.setStyleSheet(stylesheet)
        elif sys.platform == "darwin":
            # macos
            pass
        elif sys.platform == "linux" or sys.platform == "linux2":
            # linux
            pass

        self.budgetTab.move(0, 0)

        self.addTab(self.budgetTab, table1Label)

    def _setupTable2Tab(self, table2Label):
        self.failureTab = QTableWidget()
        self.failureTab.setEditTriggers(QTableWidget.NoEditTriggers)  # make cells unable to be edited
        self.failureTab.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
                                                            # column width fit to contents
        self.failureTab.setRowCount(1)
        self.failureTab.setColumnCount(3)
        self.failureTab.setHorizontalHeaderLabels(["Model Name", "Covariates", "Est. Budget"])

        header = self.failureTab.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)

        # only want to change style sheet for Windows
        # on other platforms with dark modes, creates light font on light background
        if sys.platform == "win32":
            # windows
            # provides bottom border for header
            stylesheet = "::section{Background-color:rgb(250,250,250);}"
            header.setStyleSheet(stylesheet)
        elif sys.platform == "darwin":
            # macos
            pass
        elif sys.platform == "linux" or sys.platform == "linux2":
            # linux
            pass

        self.failureTab.move(0, 0)

        self.addTab(self.failureTab, table2Label)

#############################################################

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

    def __init__(self, modelsToRun, metricNames, data, parent=None):
        """Initializes ComputeWidget class.

        Args:
            modelsToRun: List of Model objects used for estimation calculation.
            metricNames: List of metric names as strings used for estimation
                calculation.
            data: Pandas dataframe containing imported data.
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

        self.computeTask = TaskThread(modelsToRun, metricNames, data)
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
    """
    taskFinished = pyqtSignal(dict)
    modelFinished = pyqtSignal()
    nextCalculation = pyqtSignal(str)

    def __init__(self, modelsToRun, metricNames, data):
        """Initializes TaskThread class.

        Args:
            modelsToRun: List of Model objects used for estimation calculation.
            metricNames: List of metric names as strings used for estimation
                calculation.
            data: Pandas dataframe containing imported data (getData() method already
                called prior to being passed)
        """
        super().__init__()
        self.abort = False  # True when app closed, so thread stops running
        self._modelsToRun = modelsToRun
        self._metricNames = metricNames
        self._data = data

    def run(self):
        """Performs estimation for models/metrics.

        Called when thread is started.
        """
        result = {}
        for model in self._modelsToRun:
            for metricCombination in self._metricNames:
                # check if application has been closed
                if self.abort:
                    return  # get out of run method
                metricNames = ", ".join(metricCombination)
                if (metricCombination == ["None"]):
                    metricCombination = []

                m = model(data=self._data.getData(), metricNames=metricCombination)

                # this is the name used in tab 2 and tab 4 side menus
                # use shortened name
                runName = "{0} ({1})".format(m.shortName, metricNames)  # "Model (Metric1, Metric2, ...)"
                self.nextCalculation.emit(runName)

                # THIS IS WHERE SUBSETS OF COVARIATE DATA CAN BE PASSED
                # for now, just pass all
                m.runEstimation(m.covariateData)
                result[runName] = m
                self.modelFinished.emit()

        self.taskFinished.emit(result)


class PSSEThread(QThread):
    """Runs estimation calculations on separate thread.

    Attributes:
        abort: Boolean indicating if the app has been closed. If True, the
            thread should stop running.
        modelFinished: pyqtSignal, emits when current model/metric calculation
            is completed. Tells thread to begin calculation on next
            combination.
        taskFinished: pyqtSignal, emits dict containing model objects (with
            estimation results as properties) as values, indexed by name of
            model/metric combination.
        _modelsToRun: List of Model objects used for estimation calculation.
        _metricNames: List of metric names as strings used for estimation
            calculation.
        _data: Pandas dataframe containing imported data.
    """
    results = pyqtSignal(dict)

    def __init__(self, modelsToRun, metricNames, data, fraction):
        """Initializes TaskThread class.

        Args:
            modelsToRun: List of Model objects used for estimation calculation.
            metricNames: List of metric names as strings used for estimation
                calculation.
            data: Pandas dataframe containing imported data (getData() method already
                called prior to being passed)
            fraction: fraction of data to use for PSSE
        """
        super().__init__()
        self.abort = False  # True when app closed, so thread stops running
        self._modelsToRun = modelsToRun
        self._metricNames = metricNames
        self._data = data
        self._fraction = fraction

    def run(self):
        """Performs estimation for models/metrics.

        Called when thread is started.
        """
        result = {}
        for model in self._modelsToRun:
            for metricCombination in self._metricNames:
                # check if application has been closed
                if self.abort:
                    return  # get out of run method
                metricNames = ", ".join(metricCombination)
                if (metricCombination == ["None"]):
                    metricCombination = []

                m = model(data=self._data.getDataSubset(self._fraction), metricNames=metricCombination)

                # this is the name used in tab 2 and tab 4 side menus
                # use shortened name
                runName = "{0} ({1})".format(m.shortName, metricNames)  # "Model (Metric1, Metric2, ...)"

                # THIS IS WHERE SUBSETS OF COVARIATE DATA CAN BE PASSED
                # for now, just pass all
                m.runEstimation(m.covariateData)

                fitted_array = prediction_psse(m, self._data)
                psse_val = PSSE(fitted_array, self._data.getData()['CFC'].values, m.n)
                result[runName] = psse_val

        self.results.emit(result)
