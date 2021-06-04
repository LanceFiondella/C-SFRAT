# To check platform
import sys

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QGridLayout, \
                            QTableWidget, QTableWidgetItem, QAbstractScrollArea, \
                            QSpinBox, QSpacerItem, QSizePolicy, QHeaderView, QVBoxLayout, \
                            QListWidget, QAbstractItemView, QGroupBox, QListWidgetItem, \
                            QFrame, QTableView, QSlider, QDoubleSpinBox, QPushButton
from PyQt5.QtCore import pyqtSignal, QSortFilterProxyModel, Qt
from PyQt5.QtGui import QFont, QIntValidator

import pandas as pd

# For exporting table to csv
import csv

# Local imports
from core.goodnessOfFit import Comparison
from core.dataClass import PandasModel, ProxyModel


class Tab3(QWidget):
    """Contains all widgets displayed on tab 3.

    Attributes:
        sideMenu: SideMenu object holding tab 3 widgets and their signals.
        table: QTableWidget that contains the goodness-of-fit measures for each
            calculated model/metric combination.
        font: QFont object that is formatted bold. Used to set text bold for
            cells containing the highest ranked combinations, according to the
            weighting of each measure.
    """

    def __init__(self):
        """Initializes tab 3 UI elements."""
        super().__init__()
        self._setupTab3()

    def updateTableView(self, indices):
        """
        Called when model selection changes, or weighting changes.
        """        
        self.filterByIndex(indices)

        selected_data = self.tableModel.getSelected(indices)

        self.sideMenu.comparison.criticMethod_model(selected_data, self.sideMenu)

        index = 0

        # don't want to sort by string, convert to int
        indices = [int(i) for i in indices]
        indices.sort()  # need to sort indices

        for i in indices:
            # get model index
            # subtract 1 from row index, in table we display numbering starting at 0
            row = i - 1
            meanIndex = self.tableModel.index(row, self.meanColumnIndex)
            medianIndex = self.tableModel.index(row, self.medianColumnIndex)

            self.tableModel.setData(meanIndex, self.sideMenu.comparison.meanOut[index])
            self.tableModel.setData(medianIndex, self.sideMenu.comparison.medianOut[index])
            index += 1

        # causes whole table to be redrawn
        self.table.model().layoutChanged.emit()

    def filterByIndex(self, indices):
        """
        Applies filter to table model, showing only selected fitted models.
        """
        self.proxyModel.setFilterKeyColumn(0)   # filter the column of indices

        # match exactly indices[0] or indices[1] or ...
        regex_temp = '$|^'.join(indices)
        regex = '^' + regex_temp + '$'

        self.proxyModel.setFilterRegExp(regex)

    def updateModel(self, data):
        """
        Call whenever model fitting is run
        Model always contains all result data
        """

        self.sideMenu.comparison.criticMethod(data, self.sideMenu)

        rows = []
        row_index = 0
        for key, model in data.items():
            if model.converged:
                row = [
                   row_index + 1,
                   model.shortName,
                   model.metricString,
                   model.llfVal,
                   model.aicVal,
                   model.bicVal,
                   model.sseVal,
                   "",
                   self.sideMenu.comparison.meanOut[row_index],
                   self.sideMenu.comparison.medianOut[row_index]]
                rows.append(row)
                row_index += 1
        self.dataframe = pd.DataFrame(rows, columns=self.column_names)
        self.tableModel = PandasModel(self.dataframe)

        # get index of columns, used for updating critic values
        self.meanColumnIndex = self.tableModel._data.columns.get_loc("Critic (Mean)")
        self.medianColumnIndex = self.tableModel._data.columns.get_loc("Critic (Median)")


        self.proxyModel.setSourceModel(self.tableModel)

        # self.tableModel.setAllData(self.proxyModel)

    def addResultsPSSE(self, results):
        psse_values = []
        for key, val in results.items():
            psse_values.append(val)

        columnIndex = self.tableModel._data.columns.get_loc("PSSE")

        # iterate over all rows
        for row in range(self.tableModel.rowCount()):
            # get model index
            index = self.tableModel.index(row, columnIndex)

            self.tableModel.setData(index, psse_values[row])

        # causes whole table to be redrawn
        self.table.model().layoutChanged.emit()

    def _setupTab3(self):
        """Creates tab 3 widgets and adds them to layout."""
        mainLayout = QHBoxLayout()       # main layout
        self.sideMenu = SideMenu3()
        self.table = self._setupTable()
        self.font = QFont()     # allows table cells to be bold
        self.font.setBold(True)
        mainLayout.addLayout(self.sideMenu, 15)
        mainLayout.addWidget(self.table, 85)
        self.setLayout(mainLayout)

    def _setupTable(self):
        self.column_names = ["", "Model Name", "Covariates", "Log-Likelihood", "AIC", "BIC",
                             "SSE", "PSSE", "Critic (Mean)", "Critic (Median)"]
        self.dataframe = pd.DataFrame(columns=self.column_names)
        self.tableModel = PandasModel(self.dataframe)

        table = QTableView()

        table.setEditTriggers(QTableWidget.NoEditTriggers)     # make cells unable to be edited
        table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
                                                                    # column width fit to contents
        table.setSortingEnabled(True)
        header = table.horizontalHeader()
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

        # proxy model used for sorting and filtering rows
        self.proxyModel = ProxyModel()
        self.proxyModel.setSourceModel(self.tableModel)

        # table.setModel(self.tableModel)
        table.setModel(self.proxyModel)

        return table

    def exportTable(self, path):
        """
        Export table to csv
        """
        # TODO:
        # export to excel?
        # stream writing vs line by line (?), unsure which is better/faster

        # https://stackoverflow.com/questions/57419547/struggling-to-export-csv-data-from-qtablewidget
        # https://stackoverflow.com/questions/27353026/qtableview-output-save-as-csv-or-txt

        try:
            with open(path, 'w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(self.column_names)
                for row in range(self.tableModel.rowCount()):
                    rowdata = []
                    for column in range(self.tableModel.columnCount()):
                        item = self.tableModel._data.iloc[row][column]
                        if item is not None:
                            # rowdata.append(unicode(item.text()).encode('utf8'))
                            rowdata.append(str(item))
                        else:
                            rowdata.append('')
                    writer.writerow(rowdata)

        except PermissionError:
            log.warning("File permission denied.")
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setText("File permission denied")
            msgBox.setInformativeText("If there is a file with the same name ensure that it is closed.")
            msgBox.setWindowTitle("Warning")
            msgBox.exec_()


class SideMenu3(QVBoxLayout):
    """ Side menu for tab 3.
    
    Attributes:
        comparison: Comparison object that performs the calculations to
            determine which combination best fits the data.
        llfSpinBox: QSpinBox object, specifies the weighting used for the
            log-likelihood function in the comparison.
        aicSpinBox: QSpinBox object, specifies the weighting used for the
            Akaike information criterion in the comparison.
        bicSpinBox: QSpinBox object, specifies the weighting used for the
            Bayesian information criterion in the comparison.
        sseSpinBox: QSpinBox object, specifies the weighting used for the sum
            of squares error in the comparison.
        spinBoxChangedSignal: pyqtSignal, emits when any of the spin boxes for
            goodness-of-fit comparison weighting are changed.
        modelChangedSignal:
    """

    # signals
    spinBoxChangedSignal = pyqtSignal()
    modelChangedSignal = pyqtSignal(list)
    runPSSESignal = pyqtSignal(float)

    def __init__(self):
        """Initializes tab 3 side menu UI elements."""
        super().__init__()
        self._setupSideMenu()
        self.comparison = Comparison()

    def addSelectedModels(self, modelNames):
        """Adds model names to the model list widget.

        Args:
            modelNames: list of strings, name of each model to add to list
                widget.
        """
        self.modelListWidget.addItems(modelNames)

    def _setupSideMenu(self):
        """Creates side menu group boxes and adds them to the layout."""

        self.comparisonGroup = QGroupBox("Metric Weights (0-10)")
        self.comparisonGroup.setLayout(self._setupComparisonGroup())

        self.psseGroup = QGroupBox("Specify subset data for PSSE")
        self.psseGroup.setLayout(self._setupPSSEGroup())

        self.modelsGroup = QGroupBox("Select Model Results")
        # sets minumum size for side menu
        self.modelsGroup.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.modelsGroup.setLayout(self._setupModelsGroup())
        
        self.addWidget(self.comparisonGroup, 2)
        self.addWidget(self.psseGroup)
        self.addWidget(self.modelsGroup, 7)

        self.addStretch(1)

    def _setupComparisonGroup(self):
        """Creates widget containing comparison weight spin boxes.

        Returns:
            A QGridLayout containing the created comparison spin boxes and
            corresponding labels.
        """
        comparisonLayout = QGridLayout()

        # text, row, column, layout
        self._createLabel("LLF", 0, 0, comparisonLayout)
        self._createLabel("AIC", 1, 0, comparisonLayout)
        self._createLabel("BIC", 2, 0, comparisonLayout)
        self._createLabel("SSE", 3, 0, comparisonLayout)
        self._createLabel("PSSE", 4, 0, comparisonLayout)

        # minimum value, maximum value, row, column, layout
        self.llfSpinBox = self._createSpinBox(0, 10, 0, 1, comparisonLayout)
        self.aicSpinBox = self._createSpinBox(0, 10, 1, 1, comparisonLayout)
        self.bicSpinBox = self._createSpinBox(0, 10, 2, 1, comparisonLayout)
        self.sseSpinBox = self._createSpinBox(0, 10, 3, 1, comparisonLayout)
        self.psseSpinBox = self._createSpinBox(0, 10, 4, 1, comparisonLayout)
        self.psseSpinBox.setValue(0)    # want to start at 0 before PSSE is run
        self.psseSpinBox.setDisabled(True)  # disable on startup, before PSSE is run

        comparisonLayout.setColumnStretch(1, 1)

        return comparisonLayout

    def _setupPSSEGroup(self):
        """Creates widget containing comparison weight spin boxes.

        Returns:
            A QVBoxLayout containing controls for SSE parameters.
        """
        psseLayout = QVBoxLayout()
        topLayout = QHBoxLayout()

        self.psseParameterSpinBox = QDoubleSpinBox()
        self.psseParameterSpinBox.setDecimals(2)
        self.psseParameterSpinBox.setMinimum(0.01)
        self.psseParameterSpinBox.setMaximum(0.99)
        self.psseParameterSpinBox.setValue(0.9)
        self.psseParameterSpinBox.setSingleStep(0.01)

        topLayout.addWidget(self.psseParameterSpinBox, 1)

        self.psseButton = QPushButton("Run")
        self.psseButton.setDisabled(True)
        self.psseButton.clicked.connect(self._emitRunPSSESignal)

        psseLayout.addLayout(topLayout)
        psseLayout.addWidget(self.psseButton)

        return psseLayout

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

    def _createLabel(self, text, row, col, layout):
        """Creates a text label and adds it to the side menu layout.

        Args:
            text: The string of text the label displays.
            row: The row (int) of the QGroupBox widget to add the label to.
            col: The column (int) of the QGroupBox widget to add the label to.
            layout: The layout object that the label is added to.
        """
        label = QLabel(text)
        label.setSizePolicy(QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum))
        layout.addWidget(label, row, col)

    def _createSpinBox(self, minVal, maxVal, row, col, layout):
        """Creates a QSpinBox and adds it to the side menu layout.

        Current weighting values are allowed to be between 0 and 10.

        Args:
            minVal: The minimum value allowed by the spinbox (int).
            maxVal: The maximum value allowed by the spinbox (int).
            row: The row (int) of the QGroupBox widget to add the spinbox to.
            col: The column (int) of the QGroupBox widget to add the spinbox to.
            layout: The layout object that the spin box is added to.
        Returns:
            A created QSpinBox object with specified parameters.
        """
        spinBox = QSpinBox()
        spinBox.setRange(minVal, maxVal)
        spinBox.setValue(1)  # give equal weighting of 1 by default
        spinBox.valueChanged.connect(self._emitSpinBoxChangedSignal)
        layout.addWidget(spinBox, row, col)
        return spinBox

    def _emitModelChangedSignal(self):
        """
        """
        selectedModelNames = [item.text() for item in self.modelListWidget.selectedItems()]
        self.modelChangedSignal.emit(selectedModelNames)


    def _emitSpinBoxChangedSignal(self):
        """Emits signal if any goodness-of-fit spin box is changed."""
        self.spinBoxChangedSignal.emit()

    def _emitRunPSSESignal(self):
        self.runPSSESignal.emit(self.psseParameterSpinBox.value())
