# For handling debug output
import logging as log

# PyQt5 imports for UI elements
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, \
                            QGroupBox, QListWidget, QAbstractItemView, \
                            QSpinBox
from PyQt5.QtCore import pyqtSignal

# Local imports
from ui.commonWidgets import PlotWidget
# from ui.tab1 import Tab1


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
        self.failureGroup = QGroupBox("Number of Failures to Predict")
        self.modelsGroup.setLayout(self._setupModelsGroup())
        self.failureGroup.setLayout(self._setupFailureGroup())
        self.addWidget(self.modelsGroup, 7)
        self.addWidget(self.failureGroup, 1)

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

    def _setupFailureGroup(self):
        """Creates widget containing failure spin box.

        Returns:
            A QVBoxLayout containing the created failure group.
        """
        failureGroupLayout = QVBoxLayout()
        self.failureSpinBox = QSpinBox()
        self.failureSpinBox.setMinimum(0)
        self.failureSpinBox.setValue(0)
        self.failureSpinBox.valueChanged.connect(self._emitFailureChangedSignal)
        failureGroupLayout.addWidget(self.failureSpinBox)

        return failureGroupLayout

    def _emitModelChangedSignal(self):
        """Emits signal when model list widget selection changed.

        The emitted signal contains a list of the model/metric combinations
        that are currently selected.
        """
        selectedModelNames = [item.text() for item in self.modelListWidget.selectedItems()]
        log.debug("Selected models: %s", selectedModelNames)
        self.modelChangedSignal.emit(selectedModelNames)

    def _emitFailureChangedSignal(self, failures):
        """Emits signal when failure spin box changed.

        The emitted signal contains the number of future failures to predict.
        """
        self.failureChangedSignal.emit(failures)
