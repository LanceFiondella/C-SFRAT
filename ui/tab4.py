# For handling debug output
import logging as log

# PyQt5 imports for UI elements
from PyQt5.QtWidgets import QWidget, QMessageBox, QHBoxLayout, QVBoxLayout, QLabel, \
                            QGroupBox, QListWidget, QPushButton, QAbstractItemView, \
                            QFileDialog, QCheckBox, QScrollArea, QGridLayout, \
                            QTableWidget, QTableWidgetItem, QAbstractScrollArea, \
                            QSpinBox, QDoubleSpinBox
from PyQt5.QtCore import pyqtSignal

class Tab4(QWidget):

    def __init__(self):
        super().__init__()
        self.setupTab4()

    def setupTab4(self):
        self.mainLayout = QHBoxLayout() # main tab layout

        self.sideMenu = SideMenu4()
        self.mainLayout.addLayout(self.sideMenu, 25)
        self.table = self.setupTable()
        self.mainLayout.addWidget(self.table, 75)
        self.setLayout(self.mainLayout)

    def setupTable(self):
        table = QTableWidget()
        table.setEditTriggers(QTableWidget.NoEditTriggers)     # make cells unable to be edited
        table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
                                                                    # column width fit to contents
        table.setRowCount(1)
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Model Name", "Covariates", "Estimated failures"])
        table.move(0,0)

        return table

    def createHeaderLabels(self, metricNames):
        percentNames = []
        i = 0
        for name in metricNames:
            percentNames.append("%" + name)
            i += 1
        headerLabels = ["Model Name", "Covariates", "H"] + percentNames
        return headerLabels

    def addResultsToTable(self, results, data):
        """
        results = dict
            results[name] = [EffortAllocation, Model]
        """
        self.table.setSortingEnabled(False) # disable sorting while editing contents
        self.table.clear()
        self.table.setColumnCount(3 + len(data.metricNames))
        self.table.setHorizontalHeaderLabels(self.createHeaderLabels(data.metricNames))
        self.table.setRowCount(len(results))    # set row count to include all model results, 
                                                # even if not converged
        i = 0   # rows

        for key, value in results.items():
            res = value[0]
            model = value[1]

            print(res.percentages)

            self.table.setItem(i, 0, QTableWidgetItem(model.name))   # model name
            self.table.setItem(i, 1, QTableWidgetItem(model.metricString))  # model metrics
            self.table.setItem(i, 2, QTableWidgetItem("{0:.2f}".format(res.H)))
            # number of columns = number of covariates
            j = 0
            for name in model.metricNames:
                col = data.metricNameDictionary[name]
                self.table.setItem(i, 3 + col, QTableWidgetItem("{0:.2f}".format(res.percentages[j])))
                j += 1
            i += 1

                # try:
                #     c = model.metricNameDictionary[model.metricNames[j]]    # get index from metric name
                #     self.table.setItem(i, 2 + j, QTableWidgetItem(str(c)))  # 
                # except KeyError:
                #     self.table.setItem(i, )
        self.table.setRowCount(i)   # set row count to only include converged models
        self.table.resizeColumnsToContents()    # resize column width after table is edited
        self.table.setSortingEnabled(True)      # re-enable sorting after table is edited

class SideMenu4(QVBoxLayout):
    """
    Side menu for tab 4
    """

    # signals
    runAllocationSignal = pyqtSignal(list)  # starts allocation computation

    def __init__(self):
        super().__init__()
        self.setupSideMenu()

    def setupSideMenu(self):
        self.modelsGroup = QGroupBox("Select Models/Metrics for Allocation")
        self.modelsGroup.setLayout(self.setupModelsGroup())
        self.optionsGroup = QGroupBox("Allocation Parameters")
        self.optionsGroup.setLayout(self.setupOptionsGroup())
        self.setupAllocationButton()

        self.addWidget(self.modelsGroup, 75)
        self.addWidget(self.optionsGroup, 25)
        self.addWidget(self.allocationButton)

        self.addStretch(1)

    def setupModelsGroup(self):
        modelGroupLayout = QVBoxLayout()
        self.modelListWidget = QListWidget()
        modelGroupLayout.addWidget(self.modelListWidget)
        self.modelListWidget.setSelectionMode(QAbstractItemView.MultiSelection)       # able to select multiple models

        return modelGroupLayout

    def setupOptionsGroup(self):
        optionsGroupLayout = QVBoxLayout()
        optionsGroupLayout.addWidget(QLabel("Budget"))
        self.budgetSpinBox = QDoubleSpinBox()
        # self.budgetSpinBox.setMaximumWidth(200)
        self.budgetSpinBox.setRange(0.0, 999999.0)
        self.budgetSpinBox.setValue(20)
        optionsGroupLayout.addWidget(self.budgetSpinBox)
        
        optionsGroupLayout.addWidget(QLabel("Failures"))
        self.failureSpinBox = QSpinBox()
        # self.failureSpinBox.setMaximumWidth(200)
        self.failureSpinBox.setRange(1, 999999)
        optionsGroupLayout.addWidget(self.failureSpinBox)

        return optionsGroupLayout

    def setupAllocationButton(self):
        self.allocationButton = QPushButton("Run Allocation")
        self.allocationButton.setEnabled(False) # begins disabled since no model has been run yet
        # self.allocationButton.setMaximumWidth(250)
        self.allocationButton.clicked.connect(self.emitRunAllocationSignal)

    def addSelectedModels(self, modelNames):
        """
        Results on no covariates not added to list

        Args:
            modelNames (list): list of strings, name of each model
        """

        for name in modelNames:
            if " - (No covariates)" not in name:
                self.modelListWidget.addItem(name)

    def emitRunAllocationSignal(self):
        selectedCombinationNames = [item.text() for item in self.modelListWidget.selectedItems()]
        if selectedCombinationNames:
            selectedCombinationNames = [item.text() for item in self.modelListWidget.selectedItems()]
            log.info("Selected for Allocation: %s", selectedCombinationNames)
            self.runAllocationSignal.emit(selectedCombinationNames)
        else:
            log.warning("Must select at least one model/metric combination for allocation.")
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setText("No selection made for allocation")
            msgBox.setInformativeText("Please select at least one model/metric combination.")
            msgBox.setWindowTitle("Warning")
            msgBox.exec_()