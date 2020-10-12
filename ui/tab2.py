# For handling debug output
import logging as log

# PyQt5 imports for UI elements
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, \
                            QGroupBox, QListWidget, QAbstractItemView, \
                            QSpinBox, QDoubleSpinBox, QScrollArea, QLabel, \
                            QFormLayout
from PyQt5.QtCore import pyqtSignal

# Local imports
from ui.commonWidgets import PlotWidget
from ui.tab3 import Tab3, SideMenu3


class Tab2(QWidget):
    """Contains all widgets displayed on tab 2.

    Attributes:
        sideMenu: SideMenu object holding tab 2 widgets and their signals.
        plot: PlotWidget object that contains the plot for fitted data.
    """

    def __init__(self):
        super().__init__()
        self._setupTab2()

    def _setupTab2(self):
        horizontalLayout = QHBoxLayout()       # main layout
        self.sideMenu = SideMenu2()
        horizontalLayout.addLayout(self.sideMenu, 15)
        self.plot = PlotWidget()
        horizontalLayout.addWidget(self.plot, 85)
        self.setLayout(horizontalLayout)


class SideMenu2(QVBoxLayout):
    """Side menu for tab 2.

    Attributes:
        modelsGroup: QGroupBox object, contains model/metric combinations that
            converged.
        failureGroup: QGroupBox object, contains failure spin box.
        modelListWidget: QListWidget containing names of converged model/metric
            combinations.
        failureSpinBox: QSpinBox widget, specifies number of future failures
            to predict.
        modelChangedSignal: pyqtSignal, emits list of model names that are
            currently selected in the list widget.
        failureChangedSignal: pyqtSignal, emits number of failures (int) to
            predict using selected model.
    """

    # signals
    modelChangedSignal = pyqtSignal(list)   # changes based on selection of
                                            # models in tab 2
    failureChangedSignal = pyqtSignal(int)  # changes based on failure spin box
    intensityChangedSignal = pyqtSignal(float)

    def __init__(self):
        """Initializes tab 2 side menu UI elements."""
        super().__init__()
        self._setupSideMenu()

    def addSelectedModels(self, modelNames):
        """Adds model names to the model list widget.

        Args:
            modelNames: list of strings, name of each model to add to list
                widget.
        """
        self.modelListWidget.addItems(modelNames)

    def _setupSideMenu(self):
        """Creates group box widgets and adds them to layout."""
        self.modelsGroup = QGroupBox("Select Model Results")
        self.predictionGroup = QGroupBox("Predictions")
        self.modelsGroup.setLayout(self._setupModelsGroup())
        self.predictionGroup.setLayout(self._setupPredictionGroup())
        self.addWidget(self.modelsGroup, 6)
        self.addWidget(self.predictionGroup, 4)

        self.addStretch(1)

    def _setupModelsGroup(self):
        """Creates widget containing list of converged models.

        Returns:
            A QVBoxLayout containing the created model group.
        """
        modelGroupLayout = QVBoxLayout()
        self.modelListWidget = QListWidget()
        modelGroupLayout.addWidget(self.modelListWidget)
        self.modelListWidget.setSelectionMode(QAbstractItemView.MultiSelection)  # able to select multiple models
        self.modelListWidget.itemSelectionChanged.connect(self._emitModelChangedSignal)

        return modelGroupLayout

    def _setupPredictionGroup(self):
        """Creates widgets that control prediction functionality.

        Returns:
            A QVBoxLayout containing the created prediction group.
        """
        predictionGroupLayout = QVBoxLayout()

        self.scrollLayout = QVBoxLayout()
        self.scrollWidget = QWidget()
        self.scrollWidget.setLayout(self.scrollLayout)

        self.effortScrollArea = QScrollArea()
        self.effortScrollArea.setWidgetResizable(True)
        # self.effortScrollArea.resize(300, 300)
        self.effortScrollArea.setWidget(self.scrollWidget)

        self.effortSpinBoxDict = {}

        predictionGroupLayout.addWidget(QLabel("Specify Effort Per Interval"))
        predictionGroupLayout.addWidget(self.effortScrollArea, 1)

        # self.addWid("E")
        # self.addWid("F")
        # self.addWid("C")
        # self.addWid("1")
        # self.addWid("2")
        # self.addWid("3")

        # self.setupEffortList(["E", "F", "C"])


        self.failureSpinBox = QSpinBox()
        self.failureSpinBox.setMinimum(0)
        self.failureSpinBox.setValue(0)
        self.failureSpinBox.valueChanged.connect(self._emitFailureChangedSignal)
        predictionGroupLayout.addWidget(QLabel("Number of Intervals to Predict"))
        predictionGroupLayout.addWidget(self.failureSpinBox)

        self.reliabilitySpinBox = QDoubleSpinBox()
        self.reliabilitySpinBox.setMinimum(0.0)
        self.reliabilitySpinBox.setValue(0.0)
        self.reliabilitySpinBox.valueChanged.connect(self._emitIntensityChangedSignal)
        predictionGroupLayout.addWidget(QLabel("Desired Failure Intensity"))
        predictionGroupLayout.addWidget(self.reliabilitySpinBox)

        return predictionGroupLayout

    def addWid(self, name):
        hLayout = QHBoxLayout()
        hLayout.addWidget(QLabel(name), 35)
        spinBox = QDoubleSpinBox()
        hLayout.addWidget(spinBox, 65)

        self.effortSpinBoxDict[name] = spinBox

        self.scrollLayout.addLayout(hLayout)

    def updateEffortList(self, covariates):
        """
        covariates is list of covariate names
        """

        self.effortSpinBoxDict.clear()
        
        self._clearLayout(self.scrollLayout)

        for cov in range(len(covariates)):
            self.addWid(covariates[cov])

    def _clearLayout(self, layout):
        # https://stackoverflow.com/questions/4528347/clear-all-widgets-in-a-layout-in-pyqt
        # need to loop twice:
        # scoll area contains vbox layout
        # each element of vbox layout contains hbox layout
        # then, we delete all widgets for each hbox (label and spin box)
        # deleting child widgets of layout should delete layout

        # loop over all hbox layouts in main vbox layout
        while layout.count():
            # access hbox at index 0
            hbox = layout.takeAt(0)
            # loop over all widgets in hbox
            while hbox.count():
                # access widget at index 0
                child = hbox.takeAt(0)
                if child.widget():
                    # delete widget
                    child.widget().deleteLater()

        # for i in reversed(range(layout.count())):
        #     print(i)
        #     layout.itemAt(i).widget().setParent(None)

        # if layout is not None:
        #     while layout.count():
        #         item = layout.takeAt(0)
        #         widget = item.widget()
        #         if widget is not None:
        #             widget.deleteLater()
        #         else:
        #             self.clearLayout(item.layout())

    # def setupEffortList(self, covariates):
    #     """
    #     covariates is list of covariate names
    #     """
    #     num_cov = len(covariates)
    #     # self.effortScrollArea.clear()
    #     self.effortSpinBoxList.clear()
    #     for i in range(num_cov):
    #         effortSpinBox = QDoubleSpinBox()
    #         self.effortSpinBoxList.append(effortSpinBox)
    #         self.effortScrollArea.add(effortSpinBox)

    def _emitModelChangedSignal(self):
        """Emits signal when model list widget selection changed.

        The emitted signal contains a list of the model/metric combinations
        that are currently selected.
        """
        selectedModelNames = [item.text() for item in self.modelListWidget.selectedItems()]
        #log.debug("Selected models: %s", selectedModelNames)
        print("Changed Tab 2: %s", selectedModelNames)
        self.modelChangedSignal.emit(selectedModelNames)


    def _emitFailureChangedSignal(self, failures):
        """Emits signal when failure spin box changed.

        The emitted signal contains the number of future failures to predict.
        """
        self.failureChangedSignal.emit(failures)

    def _emitIntensityChangedSignal(self, intensity):
        self.intensityChangedSignal.emit(intensity)
