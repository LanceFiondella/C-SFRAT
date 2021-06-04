# For handling debug output
import logging as log

# To check platform
import sys

# PyQt5 imports for UI elements
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, \
                            QGroupBox, QListWidget, QAbstractItemView, \
                            QSpinBox, QDoubleSpinBox, QScrollArea, QLabel, \
                            QFormLayout, QHeaderView, QMessageBox
from PyQt5.QtCore import pyqtSignal

import pandas as pd

# For exporting table to csv
import csv

# Local imports
from ui.commonWidgets import PlotAndTable
from ui.tab3 import Tab3, SideMenu3
from core.dataClass import PandasModel, ProxyModel2


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
        self.plotAndTable = PlotAndTable("Plot", "Table")
        self._setupTable()
        horizontalLayout.addWidget(self.plotAndTable, 85)
        self.setLayout(horizontalLayout)

    def _setupTable(self):
        self.column_names = ["Interval"]

        # need separate dataframes for MVF and intensity values
        self.dataframeMVF = pd.DataFrame(columns=self.column_names)
        self.dataframeIntensity = pd.DataFrame(columns=self.column_names)

        # need separate models for MVF and intensity values
        self.modelMVF = PandasModel(self.dataframeMVF)
        self.modelIntensity = PandasModel(self.dataframeIntensity)

        # tableWidget is a table view
        self.plotAndTable.tableWidget.setSortingEnabled(True)
        header = self.plotAndTable.tableWidget.horizontalHeader()
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
        self.proxyModel = ProxyModel2()
        self.proxyModel.setSourceModel(self.modelMVF)

        # self.plotAndTable.tableWidget.setModel(self.tableModel)
        self.plotAndTable.tableWidget.setModel(self.proxyModel)

    def updateTableView(self, comboNums):
        """
        Called when model selection changes, or weighting changes.
        """
        self.filterByIndex(comboNums)
        self.plotAndTable.tableWidget.model().layoutChanged.emit()

    def filterByIndex(self, comboNums):
        """
        Applies filter to table model, showing only selected fitted models.
        """
        # skip 0, always want intervals column
        for i in range(1, self.proxyModel.columnCount()):
            if str(i) in comboNums:
                # model/metric combo selected, show the column
                self.plotAndTable.tableWidget.setColumnHidden(i, False)
            else:
                # not selected, hide column
                self.plotAndTable.tableWidget.setColumnHidden(i, True)

    def updateModel(self, results):
        """
        Call whenever model fitting is run
        Model always contains all result data
        """

        # lists with number of columns equal to number of combinations in results
        mvf_list = []
        intensity_list = []

        # first column is always intervals
        # get from first value in dictionary, so we don't need to know the key
        column_names = ["Interval"]
        mvf_list.append(list(results.values())[0].t)
        intensity_list.append(list(results.values())[0].t)

        # temp data frame, need to transpose afterward

        ## MVF
        # iterate over selected models
        # store intensity values and names
        for key, model in results.items():
            mvf_list.append(model.mvf_array)
            intensity_list.append(model.intensityList)
            column_names.append(key)

        temp_df = pd.DataFrame(mvf_list)
        self.dataframeMVF = temp_df.transpose()
        self.dataframeMVF.columns = column_names
        self.modelMVF.setAllData(self.dataframeMVF)

        temp_df = pd.DataFrame(intensity_list)
        self.dataframeIntensity = temp_df.transpose()
        self.dataframeIntensity.columns = column_names
        self.modelIntensity.setAllData(self.dataframeIntensity)

        self.column_names = column_names

        self.plotAndTable.tableWidget.model().layoutChanged.emit()

    def updateTable_prediction(self, prediction_list, model_names, dataViewIndex):
        # TEMPORARY, only runs when prediction spinboxes changed
        # list with number of columns equal to number of results selected
        # fc_list = []

        # first column is always intervals
        # get from first value in dictionary, so we don't need to know the key


        if len(prediction_list) > 0:
            row_df = pd.DataFrame(prediction_list)

            # need to transpose dataframe, otherwise rows and columns are swapped
            df = row_df.transpose()
            df.columns = model_names

        else:
            df = pd.DataFrame(columns=["Interval"])

        self.column_names = model_names

        # remove NaN values from dataframe
        df.fillna("", inplace=True)

        # MVF view
        if dataViewIndex == 0:
            self.modelMVF.setAllData(df)
        # intensity view
        if dataViewIndex == 1:
            self.modelIntensity.setAllData(df)

        self.plotAndTable.tableWidget.model().layoutChanged.emit()

    def setTableModel(self, dataViewIndex):
        """
        Changes table view current model

        dataViewIndex: 0 is MVF, 1 is intensity
        """

        if dataViewIndex == 0:
            self.proxyModel.setSourceModel(self.modelMVF)
        elif dataViewIndex == 1:
            self.proxyModel.setSourceModel(self.modelIntensity)

        self.plotAndTable.tableWidget.model().layoutChanged.emit()

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
    failureChangedSignal = pyqtSignal()  # changes based on failure spin box
    intensityChangedSignal = pyqtSignal(float)


    def __init__(self):
        """Initializes tab 2 side menu UI elements."""
        super().__init__()
        self._setupSideMenu()
        self.ModelsText = []

    def addSelectedModels(self, modelNames):
        """Adds model names to the model list widget.

        Args:
            modelNames: list of strings, name of each model to add to list
                widget.
        """
        self.modelListWidget.addItems(modelNames)
        self.ModelsText.clear()
        self.ModelsText = modelNames

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
        self.effortScrollArea.setWidget(self.scrollWidget)

        self.effortSpinBoxDict = {}

        predictionGroupLayout.addWidget(QLabel("Effort per Interval"))
        predictionGroupLayout.addWidget(self.effortScrollArea, 1)


        self.failureSpinBox = QSpinBox()
        self.failureSpinBox.setMinimum(0)
        self.failureSpinBox.setValue(0)
        self.failureSpinBox.setDisabled(True)   # initialize disabled, only allow changes after model fitting
        self.failureSpinBox.valueChanged.connect(self._emitFailureChangedSignal)
        predictionGroupLayout.addWidget(QLabel("Number of Intervals to Predict"))
        predictionGroupLayout.addWidget(self.failureSpinBox)

        self.reliabilitySpinBox = QDoubleSpinBox()
        self.reliabilitySpinBox.setDecimals(4)
        self.reliabilitySpinBox.setMinimum(0.0)
        self.reliabilitySpinBox.setValue(0.0)
        self.reliabilitySpinBox.setSingleStep(0.1)
        self.reliabilitySpinBox.setDisabled(True)   # initialize disabled, only allow changes after model fitting
        self.reliabilitySpinBox.valueChanged.connect(self._emitIntensityChangedSignal)
        predictionGroupLayout.addWidget(QLabel("Failure Intensity Target"))
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

    def _emitModelChangedSignal(self):
        """Emits signal when model list widget selection changed.

        The emitted signal contains a list of the model/metric combinations
        that are currently selected.
        """
        selectedModelNames = [item.text() for item in self.modelListWidget.selectedItems()]
        self.modelChangedSignal.emit(selectedModelNames)


    def _emitFailureChangedSignal(self, failures):
        """Emits signal when failure spin box changed.

        The emitted signal contains the number of future failures to predict.
        """
        # self.failureChangedSignal.emit(failures)
        self.failureChangedSignal.emit()

    def _emitIntensityChangedSignal(self, intensity):
        self.intensityChangedSignal.emit(intensity)
