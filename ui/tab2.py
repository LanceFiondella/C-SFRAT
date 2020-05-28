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
    def __init__(self):
        super().__init__()
        self.setupTab2()

    def setupTab2(self):
        self.horizontalLayout = QHBoxLayout()       # main layout
        self.sideMenu = SideMenu2()
        self.horizontalLayout.addLayout(self.sideMenu, 25)
        self.plot = PlotWidget()
        self.horizontalLayout.addWidget(self.plot, 75)
        self.setLayout(self.horizontalLayout)

class SideMenu2(QVBoxLayout):
    """
    Side menu for tab 2
    """

    # signals
    modelChangedSignal = pyqtSignal(list)    # changes based on selection of models in tab 2
    failureChangedSignal = pyqtSignal(int)   # changes based on failure spin box

    def __init__(self):
        super().__init__()
        self.setupSideMenu()

    def setupSideMenu(self):
        self.modelsGroup = QGroupBox("Select Model Results")
        self.failureGroup = QGroupBox("Number of Failures to Predict")
        self.modelsGroup.setLayout(self.setupModelsGroup())
        self.failureGroup.setLayout(self.setupFailureGroup())
        self.addWidget(self.modelsGroup)
        self.addWidget(self.failureGroup)

        self.addStretch(1)

        # signals
        # self.sheetSelect.currentIndexChanged.connect(self.emitSheetChangedSignal)

    def setupModelsGroup(self):
        modelGroupLayout = QVBoxLayout()
        self.modelListWidget = QListWidget()
        modelGroupLayout.addWidget(self.modelListWidget)
        self.modelListWidget.setSelectionMode(QAbstractItemView.MultiSelection)       # able to select multiple models
        self.modelListWidget.itemSelectionChanged.connect(self.emitModelChangedSignal)

        return modelGroupLayout

    def setupFailureGroup(self):
        failureGroupLayout = QVBoxLayout()
        self.failureSpinBox = QSpinBox()
        self.failureSpinBox.setMinimum(0)
        self.failureSpinBox.setValue(0)
        self.failureSpinBox.valueChanged.connect(self.emitFailureChangedSignal)
        failureGroupLayout.addWidget(self.failureSpinBox)

        return failureGroupLayout

    def addSelectedModels(self, modelNames):
        """


        Args:
            modelNames (list): list of strings, name of each model
        """

        #self.modelListWidget.clear()
        # modelsRan = modelDetails["modelsToRun"]
        # metricNames = modelDetails["metricNames"]

        # loadedModels = [model.name for model in modelsRan]
        # self.modelListWidget.addItems(loadedModels)

        self.modelListWidget.addItems(modelNames)

    def emitModelChangedSignal(self):
        selectedModelNames = [item.text() for item in self.modelListWidget.selectedItems()]
        log.debug("Selected models: %s", selectedModelNames)
        self.modelChangedSignal.emit(selectedModelNames)

    def emitFailureChangedSignal(self, failures):
        self.failureChangedSignal.emit(failures)