import logging as log
import os.path
import math

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
        self.STATIC_NAMES = ['T', 'FC', 'CFC']
        self.STATIC_COLUMNS = len(self.STATIC_NAMES)  # 3 for T, FC, CFC columns
        self.dataSet = {"None": None}
        # self._numCovariates = 0
        self.numCovariates = 0
        self._n = 0
        self.containsHeader = True
        self.metricNames = []
        self.metricNameCombinations = []
        self.metricNameDictionary = {}
        self._max_interval = 0
        self.setupMetricNameDictionary()

    @property
    def currentSheet(self):
        return self._currentSheet

    @currentSheet.setter
    def currentSheet(self, index):
        if index < len(self.sheetNames) and index >= 0:
            self._currentSheet = index
            log.info("Current sheet index set to %d.", index)
        else:
            self._currentSheet = 0
            log.info("Cannot set sheet to index %d since the data does not contain a sheet with that index.\
                      Sheet index instead set to 0.", index)

    @property
    def n(self):
        self._n = self.dataSet[self.sheetNames[self._currentSheet]]['FC'].size
        return self._n

    @property
    def max_interval(self):
        return self._max_interval

    @max_interval.setter
    def max_interval(self, interval):
        if interval < 5:
            self._max_interval = 5
        else:
            self._max_interval = interval

    def getData(self):
        """
        Returns dataframe corresponding to the currentSheet index
        """
        full_dataset = self.dataSet[self.sheetNames[self._currentSheet]]
        try:
            subset = full_dataset[:self._max_interval]
        except TypeError:
            # if None type, data hasn't been loaded
            # cannot subscript None type
            return full_dataset
        return subset

    def getDataSubset(self, fraction):
        """
        Returns subset of dataframe corresponding to the currentSheet index

        Args:
            percentage: float between 0.0 and 1.0 indicating percentage of
                data to return
        """
        intervals = math.floor(self.n * fraction)

        # need at least 5 data points
        if intervals < 5:
            intervals = 5

        full_dataset = self.dataSet[self.sheetNames[self._currentSheet]]
        subset = full_dataset[:intervals]

        return subset

    def getFullData(self):
        return self.dataSet[self.sheetNames[self._currentSheet]]

    def getDataModel(self):
        """
        Returns PandasModel for the current dataFrame to be displayed
        on a QTableWidget
        """
        return PandasModel(self.getData())

    def setupMetricNameDictionary(self):
        """
        For allocation table. Allows the effort allocation to be placed in correct column.
        Metric name maps to number of metric (from imported data).
        """
        i = 0
        for name in self.metricNames:
            self.metricNameDictionary[name] = i
            i += 1

    def processFT(self, data):
        """
        Processes raw FT data to fill in any gaps
        Args:
            data: Raw pandas dataframe
        Returns:
            data: Processed pandas dataframe
        """
        # failure time
        if 'FT' not in data:
            data["FT"] = data["IF"].cumsum()

        # inter failure time
        elif 'IF' not in data:
            data['IF'] = data['FT'].diff()
            data['IF'].iloc[0] = data['FT'].iloc[0]

        if 'FN' not in data:
            data['FN'] = pd.Series([i+1 for i in range(data['FT'].size)])
        return data

    def initialNumCovariates(self, data):
        """
        Calculates the number of covariates on a given sheet
        """
        numCov = len(data.columns) - self.STATIC_COLUMNS
        # log.debug("%d covariates.", self._numCovariates)
        return numCov

    def renameHeader(self, data, numCov):
        """
        Renames column headers if covariate metrics are unnamed
        """
        data.rename(columns={data.columns[0]:"Time"}, inplace=True)
        data.rename(columns={data.columns[1]:"Failures"}, inplace=True)
        for i in range(numCov):
            data.rename(columns={data.columns[i+2]:"C{0}".format(i+1)}, inplace=True)   # changed from MetricX to CX

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
                data = pd.read_excel(fname, sheet_name=None, engine="openpyxl")
            else:
                data = pd.read_excel(fname, sheet_name=None, header=None, engine="openpyxl")
        self.sheetNames = list(data.keys())
        self._currentSheet = 0
        self.setData(data)
        self.setNumCovariates()
        self._n = data[self.sheetNames[self._currentSheet]]['FC'].size
        # self.metricNames = self.dataSet[self.sheetNames[self._currentSheet]].columns.values[2:2+self.numCovariates]
        self.setMetricNames()
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
            df = pd.read_excel(fname, header=None, nrows=rows, engine="openpyxl")
            df_header = pd.read_excel(fname, nrows=rows, engine="openpyxl")
        # has a header if datatypes of loaded dataframes are different 
        header = tuple(df.dtypes) != tuple(df_header.dtypes)
        self.containsHeader = header
        return header

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
        cumulative_column = data["FC"].cumsum()  # add column for cumulative failures
        # insert cumulative column in location directly after FC
        data.insert(data.columns.get_loc("FC") + 1, 'CFC', cumulative_column)

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
                data.rename(columns={data.columns[i+2]:"Cov{0}".format(i+1)}, inplace=True)

    def setNumCovariates(self):
        """
        Sets number of covariates for each sheet
        """
        # subtract columns for failure times, number of failures, and cumulative failures
        numCov = len(self.dataSet[self.sheetNames[self._currentSheet]].columns) - self.STATIC_COLUMNS
        if numCov >= 0:
            self.numCovariates = numCov
        else:
            self.numCovariates = 0

    def setMetricNames(self):
        # column assumed to be covariate if not labeled T, FC, or CFC

        # iterate over all columns of current sheet in dataset
        names_list = []
        for (column_name, column_data) in self.dataSet[self.sheetNames[self._currentSheet]].iteritems():
            if column_name not in self.STATIC_NAMES:
                names_list.append(column_name)  # column assumed to be covariate data
        self.metricNames = names_list

    def getMetricNameCombinations(self):
        self.metricNameCombinations = []
        comb = self.powerset(self.metricNames)
        for c in comb:
            self.metricNameCombinations.append(", ".join(c))
        self.metricNameCombinations[0] = "None"

    def powerset(self, iterable):
        """ powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3) """
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


# need to edit to fit our data
class PandasModel(QtCore.QAbstractTableModel):

    # IMPORTANT:
    # Calling values method is very slow. Instead, try to index.
    # https://stackoverflow.com/questions/53838343/pyqt5-extremely-slow-scrolling-on-qtableview-with-pandas

    def __init__(self, data, parent=None):
        QtCore.QAbstractTableModel.__init__(self, parent)
        self._data = data

    def rowCount(self, parent=None):
        return len(self._data.index)
        # DO NOT use self._data.values
        # could we use self._data.rows.size ?

    def columnCount(self, parent=None):
        return len(self._data.columns)
        # return self._data.columns.size

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                # credit: https://www.learnpyqt.com/courses/model-views/qtableview-modelviews-numpy-pandas/
                # get raw value
                value = self._data.values[index.row()][index.column()]

                # Perform per-type checks and render accordingly
                if isinstance(value, float):
                    # Render float to 3 decimal places
                    return QtCore.QVariant("%.3f" % value)

                if isinstance(value, str):
                    return QtCore.QVariant("%s" % value)

                # Default
                return QtCore.QVariant(self.roundCell(self._data.iloc[index.row()][index.column()]))
                # NOT QtCore.QVariant(self.roundCell(self._data.values[index.row()][index.column()]))

        # Not valid
        return QtCore.QVariant()

    def setData(self, index, value, role=QtCore.Qt.DisplayRole):
        if index.isValid():
            self._data.iat[index.row(), index.column()] = value

            # self.dataChanged.emit(index, index)
            return True

        return False

    def roundCell(self, value):
        if isinstance(value, np.float):
            return str(round(value, ndigits=6))
        else:
            return str(value)

    def headerData(self, section, QtOrientation=QtCore.Qt.Horizontal, role=QtCore.Qt.DisplayRole):
        if QtOrientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            columnNames = list(self._data)
            return QtCore.QVariant(str(columnNames[section]))
        return QtCore.QVariant()

    def sort(self, Ncol, order):
        """
        Sort table by given column number.

        https://stackoverflow.com/questions/28660287/sort-qtableview-in-pyqt5
        """
        self.layoutAboutToBeChanged.emit()  # not sure where this conects
        data = self._data
        # self.data = self.data.sort_values(self.headers[Ncol], ascending=order == Qt.AscendingOrder)
        try:
            self._data = data.sort_values(data.columns[Ncol], ascending=order == QtCore.Qt.AscendingOrder)
        except IndexError:
            # occurs on startup, when dataframe contains no data
            pass
        self.layoutChanged.emit()

    def setAllData(self, new_data):
        """
        data is Pandas dataframe, replaces self._data
        """
        self._data = new_data
        # self.dataChanged.emit()
        self.layoutChanged.emit()

    def getSelected(self, indices):
        """
        Return selected rows based on list of indices
        """
        return self._data.loc[self._data[''].isin(indices)]

    def changeCell(self, row, column, value):
        self._data.at[row, column] = value

        # self.dataChanged.emit(row, row, [])
        self.layoutChanged.emit()

    def clear(self):
        """
        Clears all data in data frame. Used when importing new data.
        """
        self._data = pd.DataFrame()


class ProxyModel(QtCore.QSortFilterProxyModel):
    """
    Re-implement QSortFilterProxyModel to implement sorting by float/int
    """
    def __init__(self, parent=None):
        QtCore.QSortFilterProxyModel.__init__(self, parent)

    def sort(self, Ncol, order):
        self.sourceModel().sort(Ncol, order)

class ProxyModel2(QtCore.QSortFilterProxyModel):
    """
    Tab 2 table, need to filter columns
    Re-implement QSortFilterProxyModel to implement sorting by float/int
    """
    def __init__(self, parent=None):
        QtCore.QSortFilterProxyModel.__init__(self, parent)

    def sort(self, Ncol, order):
        self.sourceModel().sort(Ncol, order)

    def filterAcceptsColumn(self, source_column, source_parent):
        """
        source_column: int representing column index
        source_parent:
        """
        # print(self.sourceModel()._data.columns[0])

        if self.sourceModel()._data.columns[source_column]:
            return True
        else:
            return False
