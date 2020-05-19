# For handling debug output
import logging as log

# PyQt5 imports for UI elements
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, \
                            QGroupBox, QListWidget, QAbstractItemView
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

    def __init__(self):
        super().__init__()
        self.setupSideMenu()

    def setupSideMenu(self):
        self.modelsGroup = QGroupBox("Select Model Results")
        self.nonConvergedGroup = QGroupBox("Did Not Converge")
        self.modelsGroup.setLayout(self.setupModelsGroup())
        self.nonConvergedGroup.setLayout(self.setupNonConvergedGroup())
        self.addWidget(self.modelsGroup, 60)
        self.addWidget(self.nonConvergedGroup, 40)

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

    def setupNonConvergedGroup(self):
        nonConvergedGroupLayout = QVBoxLayout()
        self.nonConvergedListWidget = QListWidget()
        nonConvergedGroupLayout.addWidget(self.nonConvergedListWidget)

        return nonConvergedGroupLayout

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

    def addNonConvergedModels(self, nonConvergedNames):
        self.nonConvergedListWidget.addItems(nonConvergedNames)

    def emitModelChangedSignal(self):
        selectedModelNames = [item.text() for item in self.modelListWidget.selectedItems()]
        log.info("Selected models: %s", selectedModelNames)
        self.modelChangedSignal.emit(selectedModelNames)