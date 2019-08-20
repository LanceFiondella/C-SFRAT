#-----------------------------------------------------------------------#
# TODO:
# simplify ui code, make into separate methods/classes
# how will table data be entered?
#   ignore first line? only parse floats?
# what file should be the "main" file? should the main ui file
#   call the functions?
# make sure everything that needs to be is a np array, not list
# select hazard function
# MSE vs SSE?
# make some of the covariate variables global? reduce the number
#   of parameters needed to pass to methods
# checking on other datasets
# using only a subset of metrics
# making UI easier to use for someone who doesn't understand the math
# status bar?
# options selected from menubar like in SFRAT?
# protection levels
# graph should always be on same side (right)
#------------------------------------------------------------------------#

from PyQt5.QtWidgets import QMainWindow, qApp, QMessageBox, QWidget, QTabWidget, \
                            QHBoxLayout, QVBoxLayout, QTableView, QLabel, \
                            QLineEdit

# matplotlib for plots
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# estimation setting window
import estimation_setting

# math that does covariate calculations
import covariate

# global variables
import global_variables as gv

# for importing csv failure data
import csv, codecs, threading
import os


class MainWindow(QMainWindow):
    # signals


    # debug mode?
    def __init__(self, debug=False):
        '''
        description to be created at a later time
        '''
        super().__init__()

        # setup main window parameters
        self.title = "Covariate Tool"
        self.left = 10
        self.top = 10
        self.width = 1080
        self.height = 720
        self._main = MainWidget()
        self.setCentralWidget(self._main)

        # set debug mode?
        # set data?
        # signal connections?

        self.initUI()

    def closeEvent(self, event):
        '''
        description to be created at a later time
        '''
        qApp.quit()

    def initUI(self):
        '''
        description to be created at a later time
        '''
        self.setupMenu()
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.statusBar().showMessage("Ready")   # status bar?
        self.viewType = "view"                  # not sure what this means
        self.index = 0
        self.show()

    def runModels(self, modelDetails):
        '''
        description to be created at a later time
        '''
        pass

    def displayResults(self, results):
        '''
        description to be created at a later time
        '''
        pass

    def importFile(self):
        '''
        description to be created at a later time
        '''
        pass

    def changeSheet(self, index):
        '''
        description to be created at a later time
        '''
        pass

    def setDataView(self, viewType, index):
        '''
        description to be created at a later time
        '''
        pass

    def setRawDataView(self, index):
        '''
        description to be created at a later time
        '''
        pass

    def setTrendTest(self, index):
        '''
        description to be created at a later time
        '''
        pass

    def setPlotStyle(self, style='-o', plotType="step"):
        '''
        description to be created at a later time
        '''
        pass

    def updateUI(self):
        '''
        description to be created at a later time
        '''
        pass

    def setupMenu(self):
        '''
        description to be created at a later time
        '''
        pass

    def fileOpened(self):
        pass

class MainWidget(QWidget):
    '''
    description to be created at a later time
    '''
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout()
        self.tabs = Tabs()
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

class Tabs(QTabWidget):
    def __init__(self):
        super().__init__()
        self.setupTabs()

    def setupTabs(self):
        self.tab1 = Tab1()

        self.addTab(self.tab1, "Faults")

        self.resize(300, 200)

    def setupTab1(self):
        self.tab1 = QWidget()

    def setupTab2(self):
        pass

    def setupTab3(self):
        pass

    def setupTab4(self):
        pass

    def runModels(self):
        pass

class Tab1(QWidget):
    def __init__(self):
        super().__init__()
        self.setupTab1()

    def setupTab1(self):
        self.horizontalLayout = QHBoxLayout()
        self.leftVerticalLayout = QVBoxLayout()
        self.rightVerticalLayout = QVBoxLayout()

        self.tableLabel = QLabel("Imported data table")
        self.table = QTableView()
        self.leftVerticalLayout.addWidget(self.tableLabel)
        self.leftVerticalLayout.addWidget(self.table)
        self.horizontalLayout.addLayout(self.leftVerticalLayout)

        self.plotLabel = QLabel("Imported data graph")
        self.plot = QTableView()
        self.selectedFileLabel = QLabel("Selected file:")
        self.selectedFile = QLineEdit()
        self.rightVerticalLayout.addWidget(self.plotLabel)
        self.rightVerticalLayout.addWidget(self.plot)
        self.rightVerticalLayout.addWidget(self.selectedFileLabel)
        self.rightVerticalLayout.addWidget(self.selectedFile)

        self.horizontalLayout.addLayout(self.rightVerticalLayout)

        self.setLayout(self.horizontalLayout)

    def setupFigure(self):
        self.figTabs = QTabWidget()
        
