# For handling debug output
import logging as log

# PyQt5 imports for UI elements
from PyQt5.QtWidgets import QMessageBox, QWidget, QHBoxLayout, QVBoxLayout, \
                            QLabel, QGroupBox, QComboBox, QListWidget, QPushButton, \
                            QAbstractItemView, QDoubleSpinBox
from PyQt5.QtCore import pyqtSignal

# Local imports
import models
from ui.commonWidgets import PlotAndTable
from core.trendTests import *


class Tab1(QWidget):
    """Contains all widgets displayed on tab 1.


    Attributes:
        sideMenu: SideMenu object holding tab 1 widgets and their signals.
        plotAndTable: PlotAndTable object that contains the plot for imported
            data on one tab, and table containing the data in another tab.
    """

    def __init__(self):
        """Initializes tab 1 UI elements."""
        super().__init__()
        self._setupTab1()

    def _setupTab1(self):
        """Creates tab 1 widgets and adds them to layout."""
        horizontalLayout = QHBoxLayout()       # main layout

        self.sideMenu = SideMenu1()
        horizontalLayout.addLayout(self.sideMenu, 15)
        self.plotAndTable = PlotAndTable("Plot", "Table")
        horizontalLayout.addWidget(self.plotAndTable, 85)

        self.setLayout(horizontalLayout)


class SideMenu1(QVBoxLayout):
    """Side menu for tab 1.

    Attributes:
        runButton: QPushButton object, begins estimation when clicked.
        sheetSelect: QComboBox object, for selecting which sheet of spreadsheet
            of imported data to display.
        testSelect: QComboBox object, for selecting which trend test to apply
            to data.
        confidenceSpinBox: QDoubleSpinBox, for specifying the confidence
            level of the Laplace trend test.
        modelListWidget: QListWidget containing names of loaded models.
        metricListWidget: QListWidget containing names of covariate metrics
            from imported data.
        selectAllButton: QPushButton that selects all metrics in the
            metricListWidget.
        clearAllButton: QPushButton that de-selects all metrics in the
            metricListWidget.
        viewChangedSignal: pyqtSignal, emits view type (string) and view index
            (int) when view mode is changed.
        confidenceSignal: pyqtSignal, emits Laplace confidence interval (float)
            when confidence spin box changed.
        runModelSignal: pyqtSignal, emits dict of model and metric names used
            for the estimation calculation when Run Estimation button pressed.
    """

    # signals
    viewChangedSignal = pyqtSignal(str, int)    # changes based on trend test select box
                                                # for enabling/disabling confidence spin box
    confidenceSignal = pyqtSignal(float)
    runModelSignal = pyqtSignal(dict)

    def __init__(self):
        """Initializes tab 1 side menu UI elements."""
        super().__init__()
        self._setupSideMenu()
    
    def selectAll(self):
        """Selects all items in metricListWidget.
        
        Called when select all button is pressed.
        """
        self.metricListWidget.selectAll()
        self.metricListWidget.repaint()

    def clearAll(self):
        """Clears all items in metricListWidget.
        
        Called when clear all button is pressed.
        """
        self.metricListWidget.clearSelection()
        self.metricListWidget.repaint()

    def testChanged(self):
        """Emits signal indicating that the selected trend test was changed.
        
        The emitted signal contains the index of the trend test that was
        selected (0 for Laplace, 1 for running arithmetic average).
        """
        self.testSelect.setEnabled(True)
        self.confidenceSpinBox.setEnabled(True)
        # self.viewMode.setEnabled(False)
        self.viewChangedSignal.emit('trend', self.testSelect.currentIndex())

    def _setupSideMenu(self):
        """Creates group box widgets and adds them to layout."""
        sheetGroup = QGroupBox("Select Data")
        sheetGroup.setLayout(self._setupSheetGroup())
        self.addWidget(sheetGroup, 1)

        modelsGroup = QGroupBox("Select Model(s)")
        modelsGroup.setLayout(self._setupModelsGroup())
        self.addWidget(modelsGroup, 2)

        metricsGroup = QGroupBox("Select Metric(s)")
        metricsGroup.setLayout(self._setupMetricsGroup())
        self.addWidget(metricsGroup, 2)

        self.runButton = QPushButton("Run Estimation")
        self.runButton.clicked.connect(self._emitRunModelSignal)
        self.addWidget(self.runButton, 1)

        self.addStretch(1)

        # signals
        self.sheetSelect.currentIndexChanged.connect(self._emitSheetChangedSignal)   # when sheet selection changed
        self.testSelect.currentIndexChanged.connect(self.testChanged)   # when trend test selection changed

    def _setupSheetGroup(self):
        """Creates widgets for sheet selection and trend tests.
        
        Returns:
            A QVBoxLayout containing the created sheet group.
        """

        sheetGroupLayout = QVBoxLayout()

        self.sheetSelect = QComboBox()
        self.testSelect = QComboBox()   # select trend test

        trendTests = {cls.__name__: cls for
                      cls in TrendTest.__subclasses__()}
        self.testSelect.addItems([test.name for test in
                                  trendTests.values()])
        self.testSelect.setEnabled(False)   # begin disabled, showing imported
                                            # data on startup, not trend test

        self.confidenceSpinBox = QDoubleSpinBox()
        self.confidenceSpinBox.setRange(0.0, 1.0)
        self.confidenceSpinBox.setSingleStep(0.01)  # step by 0.01
        self.confidenceSpinBox.setValue(0.95)  # default value
        self.confidenceSpinBox.setDisabled(True)    # disabled on start up
        self.confidenceSpinBox.valueChanged.connect(self._emitConfidenceSignal)

        sheetGroupLayout.addWidget(QLabel("Select Sheet"))
        sheetGroupLayout.addWidget(self.sheetSelect)
        sheetGroupLayout.addWidget(QLabel("Select Trend Test"))
        sheetGroupLayout.addWidget(self.testSelect)
        sheetGroupLayout.addWidget(QLabel("Specify Laplace Confidence Level"))
        sheetGroupLayout.addWidget(self.confidenceSpinBox)

        return sheetGroupLayout

    def _setupModelsGroup(self):
        """Creates widget containing list of loaded models.

        Returns:
            A QVBoxLayout containing the created models group.
        """
        modelGroupLayout = QVBoxLayout()
        self.modelListWidget = QListWidget()
        loadedModels = [model.name for model in models.modelList.values()]
        self.modelListWidget.addItems(loadedModels)
        log.info("%d model(s) loaded: %s", len(loadedModels), loadedModels)
        self.modelListWidget.setSelectionMode(QAbstractItemView.MultiSelection)  # able to select multiple models
        modelGroupLayout.addWidget(self.modelListWidget)

        return modelGroupLayout

    def _setupMetricsGroup(self):
        """Creates widgets for selecting covariate metrics.

        Returns:
            A QVBoxLayout containing the created metrics group.
        """
        metricsGroupLayout = QVBoxLayout()
        self.metricListWidget = QListWidget()   # metric names added dynamically from data when loaded
        self.metricListWidget.setSelectionMode(QAbstractItemView.MultiSelection)     # able to select multiple metrics
        metricsGroupLayout.addWidget(self.metricListWidget)

        buttonLayout = QHBoxLayout()
        self.selectAllButton = QPushButton("Select All")
        self.clearAllButton = QPushButton("Clear All")
        self.selectAllButton.clicked.connect(self.selectAll)
        self.clearAllButton.clicked.connect(self.clearAll)
        buttonLayout.addWidget(self.selectAllButton, 50)
        buttonLayout.addWidget(self.clearAllButton, 50)
        metricsGroupLayout.addLayout(buttonLayout)

        return metricsGroupLayout

    def _emitRunModelSignal(self):
        """Emits signal that begins estimation with selected models & metrics.

        Method called when Run Estimation button is pressed. The emitted signal
        (runModelSignal) contains a dict of model names and metric names. The
        runModelSignal is only emitted if at least one model and at least one
        metric is selected.
        """
        log.info("Run button pressed.")
        # get model names as strings
        
        selectedModelNames = [item.text() for item in self.modelListWidget.selectedItems()]

        # get model classes from models folder
        modelsToRun = [model for model in models.modelList.values() if model.name in selectedModelNames]

        # get selected metric names (IMPORTANT: returned in order they were clicked)
        selectedMetricNames = [item.text().split(", ") for item in self.metricListWidget.selectedItems()]
            # split combinations

        # only emit the run signal if at least one model and at least one metric chosen
        if selectedModelNames and selectedMetricNames:
            # self.runButton.setEnabled(False)    # disable button until estimation complete
            self.runModelSignal.emit({"modelsToRun": modelsToRun,
                                      "metricNames": selectedMetricNames})
                                      
            log.info("Run models signal emitted. Models = %s, metrics = %s", selectedModelNames, selectedMetricNames)

        # if no models selected and/or no metrics selected, create message box
        # to display this warning

        elif self.modelListWidget.count() > 0 and self.metricListWidget.count() > 0:
            # data loaded but not selected
            log.warning("Must select at least one model.")
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setText("Model not selected")
            msgBox.setInformativeText("Please select at least one model and at least one metric option.")
            msgBox.setWindowTitle("Warning")
            msgBox.exec_()
        else:
            log.warning("No data found. Data must be loaded in CSV or Excel format.")
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setText("No data found")
            msgBox.setInformativeText("Please load failure data as a .csv file or an Excel workbook (.xls, xlsx).")
            msgBox.setWindowTitle("Warning")
            msgBox.exec_()

    def _emitSheetChangedSignal(self):
        """Emits signal indicating that selected sheet has changed."""
        self.viewChangedSignal.emit("sheet", self.sheetSelect.currentIndex())

    def _emitConfidenceSignal(self):
        """Emits signal indicating that the Laplace confidence level changed.

        The emitted signal contains the value that the confidence level was
        changed to, as a float.
        """
        self.confidenceSignal.emit(self.confidenceSpinBox.value())
