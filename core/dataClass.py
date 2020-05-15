import logging
import os.path

import pandas as pd
import numpy as np

# for combinations of metric names
from itertools import combinations, chain

from PyQt5 import QtCore

class Data:
    def __init__(self):
        """
        Class that stores input data.
        This class will handle data import using: Data.importFile(filename).
        Dataframes will be stored as a dictionary with sheet names as keys
            and pandas DataFrame as values
        This class will keep track of the currently selected sheet and will
            return that sheet when getData() method is called.
        """
        self.sheetNames = ["None"]
        self._currentSheet = 0
        self.dataSet = {"None": None}
        # self._numCovariates = 0
        self.numCovariates = 0
        self.containsHeader = True
        self.metricNames = []
        self.metricNameCombinations = []
        self.metricNameDictionary = {}
        self.setupMetricNameDictionary()

    @property
    def currentSheet(self):
        return self._currentSheet

    @currentSheet.setter
    def currentSheet(self, index):
        if index < len(self.sheetNames) and index >= 0:
            self._currentSheet = index
            logging.info("Current sheet index set to {0}.".format(index))
        else:
            self._currentSheet = 0
            logging.info("Cannot set sheet to index {0} since the data does not contain a sheet with that index. Sheet index instead set to 0.".format(index))

    def setupMetricNameDictionary(self):
        """
        For allocation table. Allows the effort allocation to be placed in correct column.
        Metric name maps to number of metric (from imported data).
        """
        i = 0
        for name in self.metricNames:
            self.metricNameDictionary[name] = i
            i += 1

    def setNumCovariates(self):
        # -3 columns for failure times, number of failures, and cumulative failures
        numCov = len(self.dataSet[self.sheetNames[self._currentSheet]].columns) - 3
        self.numCovariates = numCov

    def setData(self, dataSet):
        """
        Processes raw sheet data into data required by models

        failure times | number of failures | metric 1 | metric 2 | ...

        Column titles not required, data assumed to be in this format

        Args:
            dataSet : dictionary of raw data imported in importFile()
        """
        for sheet, data in dataSet.items():
            if "FC" not in data:
                raise KeyError("Column 'FC' containing failure count not found in imported file.")
            if "T" not in data:
                # data["T"] = pd.Series([i+1 for i in range(data["FC"].size)])
                data.insert(loc=0, column='T', value=pd.Series([i+1 for i in range(data['FC'].size)]))
            dataSet[sheet] = self.processRawData(data)
            numCov = self.initialNumCovariates(data)
            if self.containsHeader:
                self.metricsUnnamed(data, numCov)
            else:
                self.renameHeader(data, numCov)
        self.dataSet = dataSet

    # data not processed, currently
    def processRawData(self, data):
        """
        Add column for cumulative failures
        Args:
            data : raw pandas dataframe
        Returns:
            data : processed pandas dataframe
        """
        # print(data)
        data["Cumulative"] = data["FC"].cumsum()  # add column for cumulative failures
        data['IF'] = data['T'].diff()
        # print(data)
        return data

    def metricsUnnamed(self, data, numCov):
        """
        If data contains a header, but at least one column is unnamed.
        Renames column 1 to "Time" if unnamed,
        Renames column 2 to "Failures" if unnamed,
        Renames columns 3 through X to "MetricX" individually if unnamed
        """
        if "Unnamed: " in str(data.columns[0]):
            data.rename(columns={data.columns[0]:"Time"}, inplace=True)
        if "Unnamed: " in str(data.columns[1]):
            data.rename(columns={data.columns[1]:"Failures"}, inplace=True)
        for i in range(numCov):
            if "Unnamed: " in str(data.columns[i+2]):
                data.rename(columns={data.columns[i+2]:"Metric{0}".format(i+1)}, inplace=True)

    def initialNumCovariates(self, data):
        """
        Calculates the number of covariates on a given sheet
        """
        numCov = len(data.columns) - 3
        # logging.debug("{0} covariates.".format(self._numCovariates))
        return numCov

    def renameHeader(self, data, numCov):
        """
        Renames column headers if covariate metrics are unnamed
        """
        data.rename(columns={data.columns[0]:"Time"}, inplace=True)
        data.rename(columns={data.columns[1]:"Failures"}, inplace=True)
        for i in range(numCov):
            data.rename(columns={data.columns[i+2]:"Metric{0}".format(i+1)}, inplace=True)

    def getData(self):
        """
        Returns dataframe corresponding to the currentSheet index
        """
        return self.dataSet[self.sheetNames[self._currentSheet]]

    def getDataModel(self):
        """
        Returns PandasModel for the current dataFrame to be displayed
        on a QTableWidget
        """
        return PandasModel(self.getData())

    def importFile(self, fname):
        """
        Imports data file
        Args:
            fname : Filename of csv or excel file
        """
        self.filename, fileExtenstion = os.path.splitext(fname)
        if fileExtenstion == ".csv":
            if self.hasHeader(fname, fileExtenstion):
                # data has header, can read in normally
                data = {}
                data["None"] = pd.read_csv(fname)
            else:
                # data does not have a header, need to specify
                data = {}
                data["None"] = pd.read_csv(fname, header=None)
        else:
            if self.hasHeader(fname, fileExtenstion):
                # data has header, can read in normally
                #   *** don't think it takes into account differences in sheets
                data = pd.read_excel(fname, sheet_name=None)
            else:
                data = pd.read_excel(fname, sheet_name=None, header=None)
        self.sheetNames = list(data.keys())
        self._currentSheet = 0
        self.setData(data)
        self.setNumCovariates()
        self.metricNames = self.dataSet[self.sheetNames[self._currentSheet]].columns.values[2:-1]
        self.getMetricNameCombinations()
        self.setupMetricNameDictionary()

    def hasHeader(self, fname, extension, rows=2):
        """
        Determines if loaded data has a header
        Args:
            fname : Filename of csv or excel file
            extension : file extension of opened file
            rows : number of rows of file to compare
        Returns:
            bool : True if data has header, False if it does not
        """
        if extension == ".csv":
            df = pd.read_csv(fname, header=None, nrows=rows)
            df_header = pd.read_csv(fname, nrows=rows)
        else:
            df = pd.read_excel(fname, header=None, nrows=rows)
            df_header = pd.read_excel(fname, nrows=rows)
        # has a header if datatypes of loaded dataframes are different 
        header = tuple(df.dtypes) != tuple(df_header.dtypes)
        self.containsHeader = header
        return header

    def getMetricNameCombinations(self):
        # if (self.numCovariates > 1):
        #     self.metricNameCombinations = self.metricNames
        #     comb = []
        #     for i in range(2, self.numCovariates + 1):
        #         comb.append(list(combinations(self.metricNames,i)))
        #     print("comb =", comb)
        #     print(", ".join(comb[0][0]))

        self.metricNameCombinations = []
        comb = self.powerset(self.metricNames)
        for c in comb:
            self.metricNameCombinations.append(", ".join(c))
        # self.metricNameCombinations.remove("")  # remove option for zero metrics, user can select zero by having none selected
        self.metricNameCombinations[0] = "No covariates"
        print(self.metricNameCombinations)

    def powerset(self, iterable):
        """ powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3) """
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

# need to edit to fit our data
class PandasModel(QtCore.QAbstractTableModel):
    def __init__(self, data, parent=None):
        QtCore.QAbstractTableModel.__init__(self, parent)
        self._data = data

    def rowCount(self, parent=None):
        return len(self._data.values)
        # could we use self._data.rows.size ?

    def columnCount(self, parent=None):
        return self._data.columns.size

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                return QtCore.QVariant(self.round(self._data.values[index.row()][index.column()]))
        return QtCore.QVariant()

    def round(self, value):
        if isinstance(value, np.float):
            return str(round(value, ndigits=6))
        else:
            return str(value)

    def headerData(self, section, QtOrientation, role=QtCore.Qt.DisplayRole):
        if QtOrientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            columnNames = list(self._data)
            return QtCore.QVariant(str(columnNames[section]))
        return QtCore.QVariant()