#-----------------------------------------------------------------------------------#
# TODO:
# how will table data be entered?
#   ignore first line? only parse floats?
# make sure everything that needs to be is a np array, not list
# select hazard function
# MSE vs SSE?
# checking on other datasets
# using only a subset of metrics
# making UI easier to use for someone who doesn't understand the math
# status bar?
# options selected from menubar like in SFRAT
# protection levels
# graph should always be on same side (right)?
# setting status tips
# dialog asking if you want to quit?
# pay attention to how scaling/strecting works, minimum sizes for UI elements
# naming conventions for excel/csv
# less classes?
#   example: self._main.tabs.tab1.sideMenu.sheetSelect.addItems(self.data.sheetNames)
# figure out access modifiers, public/private variables, properties
# use logging object, removes matplotlib debug messages in debug mode
# changing sheets during calculations?
# change column header names if they don't have any
# numCovariates - property?
# do Model abstract peroperties do anything?
# figure out "if self.data.getData() is not None"
#   just need self.dataLoaded?
#------------------------------------------------------------------------------------#

# PyQt5 imports for UI elements
from PyQt5.QtWidgets import QMainWindow, qApp, QMessageBox, QWidget, QTabWidget, \
                            QHBoxLayout, QVBoxLayout, QTableView, QLabel, \
                            QLineEdit, QGroupBox, QComboBox, QListWidget, \
                            QPushButton, QAction, QActionGroup, QAbstractItemView, \
                            QFileDialog
from PyQt5.QtCore import pyqtSignal

# Matplotlib imports for graphs/plots
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# For handling debug output
import logging

# Local imports
from ui.commonWidgets import PlotWidget, PlotAndTable
from core.dataClass import Data
from core.graphSettings import PlotSettings

# math that does covariate calculations
# import covariate

# global variables
# import global_variables as gv

# for importing csv failure data
# import csv, codecs, threading
# import os


class MainWindow(QMainWindow):
    # signals
    importFileSignal = pyqtSignal()

    # debug mode?
    def __init__(self, debug=False):
        '''
        description to be created at a later time
        '''
        super().__init__()

        # setup main window parameters
        self.title = "Covariate Tool"
        self.left = 100
        self.top = 100
        self.width = 1080
        self.height = 720
        self._main = MainWidget()
        self.setCentralWidget(self._main)

        # set debug mode?
        self.debug = debug

        # set data
        self.data = Data()
        self.plotSettings = PlotSettings()

        self.dataLoaded = False

        self.ax = self._main.tabs.tab1.plotAndTable.figure.add_subplot(111)

        # signal connections
        self.importFileSignal.connect(self.importFile)
        self._main.tabs.tab1.sideMenu.viewChangedSignal.connect(self.setDataView)

        self.initUI()
        logging.info("UI loaded.")

    def closeEvent(self, event):
        '''
        description to be created at a later time
        '''
        logging.info("Covariate Tool application closed.")
        qApp.quit()

    def initUI(self):
        '''
        description to be created at a later time
        '''
        self.setupMenu()
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.statusBar().showMessage("")
        self.viewType = "view"
        self.dataViewIndex = 0
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
        Import selected file
        '''
        self._main.tabs.tab1.sideMenu.sheetSelect.clear()   # clear sheet names from previous file
        self._main.tabs.tab1.sideMenu.sheetSelect.addItems(self.data.sheetNames)    # add sheet names from new file

        self.setDataView("view", self.dataViewIndex)
        self.setMetricsList()

    def changeSheet(self, index):
        '''
        Change the current sheet displayed

        Args:
            index : index of the sheet
        '''
        self.data.currentSheet = index
        self.setDataView("view", self.dataViewIndex)
        self._main.tabs.tab1.plotAndTable.figure.canvas.draw()
        self.setMetricsList()

    def setMetricsList(self):
        self._main.tabs.tab1.sideMenu.metricsList.clear()
        if self.dataLoaded:
            dataframe = self.data.getData()
            self._main.tabs.tab1.sideMenu.metricsList.addItems(dataframe.columns.values[2:-1])

    def setDataView(self, viewType, index):
        '''
        Set the data to be displayed. 
        Called whenever a menu item is changed

        Args:
            viewType: string that determines view
            index: index of the dataview list
        '''
        if self.data.getData() is not None:
            if viewType == "view":
                self.setRawDataView(index)
            elif viewType == "trend":
                self.setTrendTest(index)
            elif viewType == "sheet":
                self.changeSheet(index)
            self.viewType = viewType
            self.dataViewIndex = index

    def setRawDataView(self, index):
        '''
        Changes plot between MVF and intensity
        '''
        self._main.tabs.tab1.plotAndTable.tableWidget.setModel(self.data.getDataModel())
        dataframe = self.data.getData()
        if index == 0:
            # MVF
            self.ax = self.plotSettings.generatePlot(self.ax, dataframe.iloc[:, 0], dataframe["Cumulative"], title="MVF", xLabel="time", yLabel="failures")
        if index == 1:
            # Intensity
            self.ax = self.plotSettings.generatePlot(self.ax, dataframe.iloc[:, 0], dataframe.iloc[:, 1], title="Intensity", xLabel="time", yLabel="failures")

        self._main.tabs.tab1.plotAndTable.figure.canvas.draw()

    def setTrendTest(self, index):
        '''
        description to be created at a later time
        '''
        pass

    def setPlotStyle(self, style='-o', plotType="step"):
        '''
        description to be created at a later time
        '''
        self.plotSettings.style = style
        self.plotSettings.plotType = plotType
        self.updateUI()

    def updateUI(self):
        '''
        Change Plot, Table and SideMenu
        when the state of the Data object changes

        Should be called explicitly
        '''
        self.setDataView(self.viewType, self.dataViewIndex)

    def setupMenu(self):
        '''
        description to be created at a later time
        '''
        self.menu = self.menuBar()      # initialize menu bar

        # ---- File menu
        fileMenu = self.menu.addMenu("File")
        # open
        openFile = QAction("Open", self)
        openFile.setShortcut("Ctrl+O")
        openFile.setStatusTip("Import Data File")
        openFile.triggered.connect(self.fileOpened)
        # exit
        exitApp = QAction("Exit", self)
        exitApp.setShortcut("Ctrl+Q")
        exitApp.setStatusTip("Close application")
        exitApp.triggered.connect(self.closeEvent)
        # add actions to file menu
        fileMenu.addAction(openFile)
        fileMenu.addSeparator()
        fileMenu.addAction(exitApp)

        # ---- View menu
        viewMenu = self.menu.addMenu("View")
        # -- plotting style
        # maybe want a submenu?
        viewStyle = QActionGroup(viewMenu)
        # points
        viewPoints = QAction("Show Points", self, checkable=True)
        viewPoints.setShortcut("Ctrl+P")
        viewPoints.setStatusTip("Data shown as points on graphs")
        viewPoints.triggered.connect(self.setPointsView)
        viewStyle.addAction(viewPoints)
        # lines
        viewLines = QAction("Show Lines", self, checkable=True)
        viewLines.setShortcut("Ctrl+L")
        viewLines.setStatusTip("Data shown as lines on graphs")
        viewLines.triggered.connect(self.setLineView)
        viewStyle.addAction(viewLines)
        # points and lines
        viewBoth = QAction("Show Points and Lines", self, checkable=True)
        viewBoth.setShortcut("Ctrl+B")
        viewBoth.setStatusTip("Data shown as points and lines on graphs")
        viewBoth.setChecked(True)
        viewBoth.triggered.connect(self.setLineAndPointsView)
        viewStyle.addAction(viewBoth)
        # add actions to view menu
        viewMenu.addActions(viewStyle.actions())
        # -- graph display
        graphStyle = QActionGroup(viewMenu)
        # MVF
        mvf = QAction("MVF Graph", self, checkable=True)
        mvf.setShortcut("Ctrl+M")
        mvf.setStatusTip("Graphs display MVF of data")
        mvf.setChecked(True)
        mvf.triggered.connect(self.setMVFView)
        graphStyle.addAction(mvf)
        # intensity
        intensity = QAction("Intensity Graph", self, checkable=True)
        intensity.setShortcut("Ctrl+I")
        intensity.setStatusTip("Graphs display failure intensity")
        intensity.triggered.connect(self.setIntensityView)
        graphStyle.addAction(intensity)
        # add actions to view menu
        viewMenu.addSeparator()
        viewMenu.addActions(graphStyle.actions())

    #region Menu actions
    def fileOpened(self):
        files = QFileDialog.getOpenFileName(self, "Open profile", "", filter=("Data Files (*.csv *.xls *.xlsx)"))
        # if a file was loaded
        if files[0]:
            self.data.importFile(files[0])      # imports loaded file
            self.dataLoaded = True
            logging.info("Data loaded from {0}".format(files[0]))
            self.importFileSignal.emit()            # emits signal that file was imported successfully

    def setLineView(self):
        self.setPlotStyle(style='-')
        logging.info("Plot style set to line view.")

    def setPointsView(self):
        self.setPlotStyle(style='o', plotType='plot')
        logging.info("Plot style set to points view.")
    
    def setLineAndPointsView(self):
        self.setPlotStyle(style='-o')
        logging.info("Plot style set to line and points view.")

    def setMVFView(self):
        self.dataViewIndex = 0
        logging.info("Data plots set to MVF view.")
        if self.dataLoaded:
            self.setRawDataView(self.dataViewIndex)

    def setIntensityView(self):
        self.dataViewIndex = 1
        logging.info("Data plots set to intensity view.")
        if self.dataLoaded:
            self.setRawDataView(self.dataViewIndex)
    #endregion


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
        self.addTab(self.tab1, "Data Upload and Model Selection")

        self.tab2 = Tab2()
        self.addTab(self.tab2, "Model Results and Predictions")

        self.tab3 = Tab3()
        self.addTab(self.tab3, "Model Comparison")

        self.resize(300, 200)

    # def setupTab1(self):
    #     self.tab1 = QWidget()

    # def setupTab2(self):
    #     pass

    # def setupTab3(self):
    #     pass

    def runModels(self):
        pass

#region Tab 1
class Tab1(QWidget):
    def __init__(self):
        super().__init__()
        self.setupTab1()

    def setupTab1(self):
        self.horizontalLayout = QHBoxLayout()       # main layout

        self.sideMenu = SideMenu()
        self.horizontalLayout.addLayout(self.sideMenu, 25)

        # self.plotGroup = QGroupBox("Plot and Table of Imported Data")
        # self.plotGroup.setLayout(self.setupPlotGroup())
        self.plotAndTable = PlotAndTable()
        self.horizontalLayout.addWidget(self.plotAndTable, 75)

        self.setLayout(self.horizontalLayout)

    # def setupPlotGroup(self):
    #     plotAndTableLayout = QVBoxLayout()
    #     self.plotAndTable = PlotAndTable()
    #     plotAndTableLayout.addWidget(self.plotAndTable)
    #     return plotAndTableLayout

class SideMenu(QVBoxLayout):
    '''
    Side menu for tab 1
    '''

    # signals
    viewChangedSignal = pyqtSignal(str, int)    # should this be before init?

    def __init__(self):
        super().__init__()
        self.setupSideMenu()

    def setupSideMenu(self):
        self.sheetGroup = QGroupBox("Select Sheet")
        self.sheetGroup.setLayout(self.setupSheetGroup())
        self.addWidget(self.sheetGroup)

        self.modelsGroup = QGroupBox("Select Model(s)")
        self.modelsGroup.setLayout(self.setupModelsGroup())
        self.addWidget(self.modelsGroup)

        self.metricsGroup = QGroupBox("Select Metric(s)")
        self.metricsGroup.setLayout(self.setupMetricsGroup())
        self.addWidget(self.metricsGroup)

        self.runButton = QPushButton("Run Estimation")
        self.addWidget(self.runButton)

        self.addStretch(1)

        # signals
        self.sheetSelect.currentIndexChanged.connect(self.sheetChanged)

    def setupSheetGroup(self):
        sheetGroupLayout = QVBoxLayout()
        # sheetGroupLayout.addWidget(QLabel("Select sheet"))

        self.sheetSelect = QComboBox()
        sheetGroupLayout.addWidget(self.sheetSelect)

        return sheetGroupLayout

    def setupModelsGroup(self):
        modelGroupLayout = QVBoxLayout()
        self.modelList = QListWidget()

        # TEMPORARY
        # will later dynamically add model names
        self.modelList.addItems(["Geometric", "Negative Binomial (Order 2)", "Discrete Weibull (Order 2)"])
        self.modelList.setSelectionMode(QAbstractItemView.MultiSelection)       # able to select multiple models
        modelGroupLayout.addWidget(self.modelList)

        return modelGroupLayout

    def setupMetricsGroup(self):
        metricsGroupLayout = QVBoxLayout()
        self.metricsList = QListWidget()

        # TEMPORARY
        # will later dynamically add metric names (if given)
        # self.metricsList.addItems(["Metric 1", "Metric 2", "Metric 3"])
        self.metricsList.setSelectionMode(QAbstractItemView.MultiSelection)     # able to select multiple metrics
        metricsGroupLayout.addWidget(self.metricsList)

        return metricsGroupLayout

    def sheetChanged(self):
        self.viewChangedSignal.emit("sheet", self.sheetSelect.currentIndex())
#endregion

#region Tab 2
class Tab2(QWidget):
    def __init__(self):
        super().__init__()
        self.setupTab2()

    def setupTab2(self):
        self.horizontalLayout = QHBoxLayout()       # main layout

        self.plotGroup = QGroupBox("Model Results")
        self.plotGroup.setLayout(self.setupPlotGroup())
        self.horizontalLayout.addWidget(self.plotGroup, 60)

        self.tableGroup = QGroupBox("Table of Predictions")
        self.tableGroup.setLayout(self.setupTableGroup())
        self.horizontalLayout.addWidget(self.tableGroup, 40)

        self.setLayout(self.horizontalLayout)

    def setupPlotGroup(self):
        plotLayout = QVBoxLayout()
        self.plot = PlotWidget()
        plotLayout.addWidget(self.plot)
        return plotLayout

    def setupTableGroup(self):
        tableLayout = QVBoxLayout()
        self.table = QTableView()
        tableLayout.addWidget(self.table)
        return tableLayout
#endregion

#region Tab 3
class Tab3(QWidget):
    def __init__(self):
        super().__init__()
        self.setupTab3()

    def setupTab3(self):
        self.horizontalLayout = QHBoxLayout()       # main layout

        self.table = QTableView()
        self.horizontalLayout.addWidget(self.table)

        self.setLayout(self.horizontalLayout)
#endregion