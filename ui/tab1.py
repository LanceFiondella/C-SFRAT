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
    def __init__(self):
        super().__init__()
        self.setupTab1()

    def setupTab1(self):
        self.horizontalLayout = QHBoxLayout()       # main layout

        self.sideMenu = SideMenu1()
        self.horizontalLayout.addLayout(self.sideMenu, 25)
        self.plotAndTable = PlotAndTable("Plot", "Table")
        self.horizontalLayout.addWidget(self.plotAndTable, 75)

        self.setLayout(self.horizontalLayout)

class SideMenu1(QVBoxLayout):
    """
    Side menu for tab 1
    """

    # signals
    viewChangedSignal = pyqtSignal(str, int)
    runModelSignal = pyqtSignal(dict)


    def __init__(self):
        super().__init__()
        self.setupSideMenu()

    def setupSideMenu(self):
        self.sheetGroup = QGroupBox("Select Data")
        self.sheetGroup.setLayout(self.setupSheetGroup())
        self.addWidget(self.sheetGroup)

        self.modelsGroup = QGroupBox("Select Model(s)")
        self.modelsGroup.setLayout(self.setupModelsGroup())
        self.addWidget(self.modelsGroup)

        self.metricsGroup = QGroupBox("Select Metric(s)")
        self.metricsGroup.setLayout(self.setupMetricsGroup())
        self.addWidget(self.metricsGroup)

        self.runButton = QPushButton("Run Estimation")
        self.runButton.clicked.connect(self.emitRunModelsSignal)
        self.addWidget(self.runButton)

        self.addStretch(1)

        # signals
        self.sheetSelect.currentIndexChanged.connect(self.emitSheetChangedSignal)   # when sheet selection changed
        self.testSelect.currentIndexChanged.connect(self.testChanged)   # when trend test selection changed

    def setupSheetGroup(self):
        sheetGroupLayout = QVBoxLayout()
        # sheetGroupLayout.addWidget(QLabel("Select sheet"))

        self.sheetSelect = QComboBox()
        self.testSelect = QComboBox()   # select trend test
        
        trendTests = {cls.__name__: cls for
                      cls in TrendTest.__subclasses__()}
        self.testSelect.addItems([test.name for test in
                                  trendTests.values()])
        self.testSelect.setEnabled(False)   # begin disabled, showing imported data on startup, not trend test

        self.confidenceSpinBox = QDoubleSpinBox()
        self.confidenceSpinBox.setRange(0.0, 1.0)
        self.confidenceSpinBox.setSingleStep(0.01)  # step by 0.01
        self.confidenceSpinBox.setValue(0.95) # default value
        self.confidenceSpinBox.setDisabled(True)    # disabled on start up

        sheetGroupLayout.addWidget(QLabel("Select Sheet"))
        sheetGroupLayout.addWidget(self.sheetSelect)
        sheetGroupLayout.addWidget(QLabel("Select Trend Test"))
        sheetGroupLayout.addWidget(self.testSelect)
        sheetGroupLayout.addWidget(QLabel("Specify Laplace Confidence Level"))
        sheetGroupLayout.addWidget(self.confidenceSpinBox)

        return sheetGroupLayout

    def setupModelsGroup(self):
        modelGroupLayout = QVBoxLayout()
        self.modelListWidget = QListWidget()
        loadedModels = [model.name for model in models.modelList.values()]
        self.modelListWidget.addItems(loadedModels)
        log.info("%d model(s) loaded: %s", len(loadedModels), loadedModels)
        self.modelListWidget.setSelectionMode(QAbstractItemView.MultiSelection)       # able to select multiple models
        modelGroupLayout.addWidget(self.modelListWidget)

        return modelGroupLayout

    def setupMetricsGroup(self):
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
    
    def selectAll(self):
        self.metricListWidget.selectAll()
        self.metricListWidget.repaint()

    def clearAll(self):
        self.metricListWidget.clearSelection()
        self.metricListWidget.repaint()

    def emitRunModelsSignal(self):
        """
        Method called when Run Estimation button pressed.
        Signal that tells models to run (runModelSignal) is
        only emitted if at least one model and at least one
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
        # sorts metric names in their order from the data file (left to right)
        #metricNames = [self.metricListWidget.item(i).text() for i in range(self.metricListWidget.count()) if self.metricListWidget.item(i).text() in selectedMetricNames]
        # only emit the run signal if at least one model and at least one metric chosen
        if selectedModelNames and selectedMetricNames:
            # self.runButton.setEnabled(False)    # disable button until estimation complete
            self.runModelSignal.emit({"modelsToRun": modelsToRun,
                                      "metricNames": selectedMetricNames})
                                      #"metricNames": metricNames})
                                      
            log.info("Run models signal emitted. Models = %s, metrics = %s", selectedModelNames, selectedMetricNames)
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

    def emitSheetChangedSignal(self):
        self.viewChangedSignal.emit("sheet", self.sheetSelect.currentIndex())

    def testChanged(self):
        self.testSelect.setEnabled(True)
        self.confidenceSpinBox.setEnabled(True)
        # self.viewMode.setEnabled(False)
        self.viewChangedSignal.emit('trend', self.testSelect.currentIndex())