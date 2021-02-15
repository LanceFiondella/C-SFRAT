# For handling debug output
import logging as log

# PyQt5 imports for UI elements
from PyQt5.QtWidgets import QMessageBox, QWidget, QHBoxLayout, QVBoxLayout, \
                            QLabel, QGroupBox, QComboBox, QListWidget, QPushButton, \
                            QAbstractItemView, QDoubleSpinBox, QSlider
from PyQt5.QtCore import pyqtSignal, Qt

# Local imports
import models
from ui.commonWidgets import PlotAndTable


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
    sliderSignal = pyqtSignal(int)

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

    def updateSlider(self, max_value):
        """
        Called when new data is imported/sheet changed. Updates slider to
        include all data points.
        """
        self.slider.setMaximum(max_value)
        self.slider.setValue(max_value)
        self.sliderLabel.setText(str(max_value))

    def _setupSideMenu(self):
        """Creates group box widgets and adds them to layout."""
        dataGroup = QGroupBox("Select Data")
        dataGroup.setLayout(self._setupDataGroup())
        self.addWidget(dataGroup, 1)

        modelsGroup = QGroupBox("Select Hazard Functions")
        modelsGroup.setLayout(self._setupModelsGroup())
        self.addWidget(modelsGroup, 2)

        metricsGroup = QGroupBox("Select Covariates")
        metricsGroup.setLayout(self._setupMetricsGroup())
        self.addWidget(metricsGroup, 2)

        self.runButton = QPushButton("Run Estimation")
        self.runButton.clicked.connect(self._emitRunModelSignal)
        self.addWidget(self.runButton, 1)

        self.addStretch(1)

        # signals
        self.sheetSelect.currentIndexChanged.connect(self._emitSheetChangedSignal)   # when sheet selection changed

    def _setupDataGroup(self):
        """Creates widgets for sheet selection and trend tests.
        
        Returns:
            A QVBoxLayout containing the created sheet group.
        """

        dataGroupLayout = QVBoxLayout()

        self.sheetSelect = QComboBox()

        sliderLayout = QHBoxLayout()

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(1)
        self.slider.setMinimum(1)
        self.slider.setMaximum(1)
        self.slider.valueChanged.connect(self._emitSliderSignal)

        self.sliderLabel = QLabel("")

        sliderLayout.addWidget(self.slider, 9)
        sliderLayout.addWidget(self.sliderLabel, 1)

        dataGroupLayout.addWidget(QLabel("Select Sheet"))
        dataGroupLayout.addWidget(self.sheetSelect)
        dataGroupLayout.addWidget(QLabel("Subset Failure Data"))
        dataGroupLayout.addLayout(sliderLayout)

        return dataGroupLayout

    def _setupModelsGroup(self):
        """Creates widget containing list of loaded models.

        Returns:
            A QVBoxLayout containing the created models group.
        """
        modelGroupLayout = QVBoxLayout()
        self.modelListWidget = QListWidget()
        loadedModels = [model.name for model in models.modelList.values()]
        self.modelListWidget.addItems(loadedModels)

        # set minimum size for list widget depending on contents
        self.modelListWidget.setMinimumWidth(300)

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

    def _emitSliderSignal(self):
        self.sliderLabel.setText(str(self.slider.value()))
        self.sliderSignal.emit(self.slider.value())
