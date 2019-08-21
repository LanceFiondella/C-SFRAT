
import logging as log
import os.path

import pandas as pd
import numpy as np

from PyQt5 import QtCore

class Data:
    def __init__(self):
        '''
        Class that stores input data.
        This class will handle data import using: Data.importFile(filename).
        Dataframes will be stored as a dictionary with sheet names as keys
            and pandas DataFrame as values
        This class will keep track of the currently selected sheet and will
            return that sheet when getData() method is called.
        '''
        self.sheetNames = ["None"]
        self._currentSheet = 0
        self.dataSet = {"None": None}

    @property
    def currentSheet(self):
        return self._currentSheet

    @currentSheet.setter
    def currentSheet(self, index):
        if index < len(self.sheetNames) and index >= 0:
            self._currentSheet = index
            log.info("Current sheet index set to {0}".format(index))
        else:
            self._currentSheet = 0
            log.info("Cannot set sheet to index {0} since the data does not contain a sheet with that index. Sheet index instead set to 0.".format(index))

    def setData(self, dataSet):
        '''
        Processes raw sheet data into data required by models

        failure times | number of failures | metric 1 | metric 2 | ...
            * names of metrics MUST be strings (cannot contain exclusively numbers) *

        Column titles not required, data assumed to be in this format

        Args:
            dataSet : dictionary of raw data imported in importFile()
        '''
        for sheet, data in dataSet.items():
            print(sheet)
            print(data.info())
            # dataSet[sheet] = self.processRawData(data)
            dataSet[sheet] = data
        self.dataSet = dataSet
        print(self.dataSet)

    # data not processed, currently
    def processRawData(self, data):
        '''
        Process raw data
        Args:
            data : raw pandas dataframe
        Returns:
            data : processed pandas dataframe
        '''
        print(data)
        data.iloc[:, 0] = pd.to_numeric(data.iloc[:, 0], errors="ignore")
        print(data)
        return data

    def getData(self):
         '''
         Returns dataframe corresponding to the currentSheet index
         '''
         return self.dataSet[self.sheetNames[self._currentSheet]]

    def importFile(self, fname):
        '''
        Imports data file
        Args:
            fname : Filename of csv or excel file
        '''
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

    def hasHeader(self, fname, extension, rows=2):
        '''
        Determines if loaded data has a header
        Args:
            fname : Filename of csv or excel file
            extension : file extension of opened file
            rows : number of rows of file to compare
        Returns:
            bool : True if data has header, False if it does not
        '''
        if extension == ".csv":
            df = pd.read_csv(fname, header=None, nrows=rows)
            df_header = pd.read_csv(fname, nrows=rows)
        else:
            df = pd.read_excel(fname, header=None, nrows=rows)
            df_header = pd.read_excel(fname, nrows=rows)
        # has a header if datatypes of loaded dataframes are different 
        header = tuple(df.dtypes) != tuple(df_header.dtypes)
        log.debug("Data table has header.")
        return header


class PandasModel(QtCore.QAbstractTableModel):
    def __init__(self, data, parent=None):
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