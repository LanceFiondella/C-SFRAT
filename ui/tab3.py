from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QGridLayout, \
                            QTableWidget, QTableWidgetItem, QAbstractScrollArea, \
                            QSpinBox, QSpacerItem, QSizePolicy
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QFont

# Local imports
from core.comparison import Comparison

class Tab3(QWidget):
    def __init__(self):
        super().__init__()
        self.setupTab3()

    def setupTab3(self):
        self.mainLayout = QHBoxLayout()       # main layout
        self.sideMenu = SideMenu3()
        self.setupTable()
        self.mainLayout.addLayout(self.sideMenu, 15)
        self.mainLayout.addWidget(self.table, 85)
        self.setLayout(self.mainLayout)

    def setupTable(self):
        self.table = QTableWidget()
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)     # make cells unable to be edited
        self.table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
                                                                    # column width fit to contents
        self.table.setRowCount(1)
        columnLabels = ["Model Name", "Covariates", "Log-Likelihood", "AIC", "BIC",
                        "SSE", "Model ranking (no weights)", "Model ranking (user-specified weights)"]
        self.table.setColumnCount(len(columnLabels))
        self.table.setHorizontalHeaderLabels(columnLabels)
        self.table.move(0,0)

        self.font = QFont() # allows table cells to be bold
        self.font.setBold(True)

    def addResultsToTable(self, results):
        # numResults = len(results)
        self.table.setSortingEnabled(False) # disable sorting while editing contents
        self.table.clear()
        self.table.setHorizontalHeaderLabels(["Model Name", "Covariates", "Log-Likelihood", "AIC", "BIC",
                                              "SSE", "Model ranking (no weights)", "Model ranking (user-specified weights)"])
                                              #"Weighted selection (mean)", "Weighted selection (median)"])
        self.table.setRowCount(len(results))    # set row count to include all model results, 
                                                # even if not converged
        i = 0   # number of converged models

        self.sideMenu.comparison.goodnessOfFit(results, self.sideMenu)

        for key, model in results.items():
            if model.converged:
                self.table.setItem(i, 0, QTableWidgetItem(model.name))
                self.table.setItem(i, 1, QTableWidgetItem(model.metricString))
                self.table.setItem(i, 2, QTableWidgetItem("{0:.3f}".format(model.llfVal)))
                self.table.setItem(i, 3, QTableWidgetItem("{0:.3f}".format(model.aicVal)))
                self.table.setItem(i, 4, QTableWidgetItem("{0:.3f}".format(model.bicVal)))
                self.table.setItem(i, 5, QTableWidgetItem("{0:.3f}".format(model.sseVal)))
                self.table.setItem(i, 6, QTableWidgetItem("{0:.3f}".format(self.sideMenu.comparison.meanOutUniform[i])))
                self.table.setItem(i, 7, QTableWidgetItem("{0:.3f}".format(self.sideMenu.comparison.meanOut[i])))
                i += 1
        self.table.setRowCount(i)   # set row count to only include converged models
        self.table.resizeColumnsToContents()    # resize column width after table is edited
        self.table.setSortingEnabled(True)      # re-enable sorting after table is edited

        self.table.item(self.sideMenu.comparison.bestMeanUniform, 6).setFont(self.font)
        self.table.item(self.sideMenu.comparison.bestMean, 7).setFont(self.font)

class SideMenu3(QGridLayout):
    """
    Side menu for tab 3
    """

    # signals
    comboBoxChangedSignal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setupSideMenu()
        self.comparison = Comparison()

    def setupSideMenu(self):
        self.createLabel("Metric", 0, 0)
        self.createLabel("weights (0-10)", 0, 1)
        self.createLabel("LLF", 1, 0)
        self.createLabel("AIC", 2, 0)
        self.createLabel("BIC", 3, 0)
        self.createLabel("SSE", 4, 0)
        self.llfSpinBox = self.createSpinBox(0, 10, 1, 1)
        self.aicSpinBox = self.createSpinBox(0, 10, 2, 1)
        self.bicSpinBox = self.createSpinBox(0, 10, 3, 1)
        self.sseSpinBox = self.createSpinBox(0, 10, 4, 1)

        # vertical spacer at bottom of layout, keeps labels/spinboxes together at top of window
        vspacer = QSpacerItem(20, 40, QSizePolicy.Maximum, QSizePolicy.Expanding)
        self.addItem(vspacer, 5, 0, 1, -1)
        self.setColumnStretch(1, 1)

    def createLabel(self, text, row, col):
        label = QLabel(text)
        label.setSizePolicy(QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum))
        self.addWidget(label, row, col)

    def createSpinBox(self, minVal, maxVal, row, col):
        spinBox = QSpinBox()
        spinBox.setRange(minVal, maxVal)
        spinBox.setValue(1) # give equal weighting of 1 by default
        spinBox.valueChanged.connect(self.emitComboBoxChangedSignal)
        self.addWidget(spinBox, row, col)
        return spinBox

    def emitComboBoxChangedSignal(self):
        self.comboBoxChangedSignal.emit()