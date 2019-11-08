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
#   think the solution is to load the data once, and then continue using that same data
#   until calculations are completed
# change column header names if they don't have any
# numCovariates - property?
# do Model abstract peroperties do anything?
# figure out "if self.data.getData() is not None"
#   just need self.dataLoaded?
# more descriptions of what's happening as estimations are running (ComputeWidget)
# predict points? (commonWidgets)
# naming "hazard functions" instead of models
# fsolve doesn't return if converged, so it's not updated for models
#   should try other scipy functions
# make tab 2 like tab 1
#   side menu with plot/table on right
#   definitely need a side menu to select the hazard functions
# names of tabs in tab 2?
# self.viewType is never updated, we don't use updateUI()
# sometimes metric list doesn't load until interacted with
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
from ui.commonWidgets import PlotWidget, PlotAndTable, ComputeWidget, TaskThread
from core.dataClass import Data
from core.graphSettings import PlotSettings
import models


class MainWindow(QMainWindow):
    # signals
    importFileSignal = pyqtSignal()

    # debug mode?
    def __init__(self, debug=False):
        """
        description to be created at a later time
        """
        super().__init__()

        # setup main window parameters
        self.title = "Covariate Tool"
        self.left = 100
        self.top = 100
        self.width = 1080
        self.height = 720
        self.minWidth = 800
        self.minHeight = 600
        self._main = MainWidget()
        self.setCentralWidget(self._main)

        # set debug mode?
        self.debug = debug

        # set data
        self.data = Data()
        self.plotSettings = PlotSettings()
        self.selectedModelNames = []

        # self.estimationResults
        # self.currentModel

        # flags
        self.dataLoaded = False
        self.estimationComplete = False

        # tab 1 plot and table
        self.ax = self._main.tabs.tab1.plotAndTable.figure.add_subplot(111)
        # tab 2 plot and table
        self.ax2 = self._main.tabs.tab2.plotAndTable.figure.add_subplot(111)

        # signal connections
        self.importFileSignal.connect(self.importFile)
        self._main.tabs.tab1.sideMenu.viewChangedSignal.connect(self.setDataView)
        self._main.tabs.tab1.sideMenu.runModelSignal.connect(self.runModels)    # run models when signal is received
        self._main.tabs.tab1.sideMenu.runModelSignal.connect(self._main.tabs.tab2.sideMenu.addSelectedModels)    # fill tab 2 models group with selected models
        self._main.tabs.tab2.sideMenu.modelChangedSignal.connect(self.changePlot2)
        # connect tab2 list changed to refreshing tab 2 plot

        self.initUI()
        logging.info("UI loaded.")

    def closeEvent(self, event):
        """
        description to be created at a later time
        """
        logging.info("Covariate Tool application closed.")
        qApp.quit()

    def initUI(self):
        """
        description to be created at a later time
        """
        self.setupMenu()
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setMinimumSize(self.minWidth, self.minHeight)
        self.statusBar().showMessage("")
        self.viewType = "view"
        self.dataViewIndex = 0
        self.show()

    def runModels(self, modelDetails):
        """
        Run selected models using selected metrics

        Args:
            modelDetails : dictionary of models and metrics to use for calculations
        """
        modelsToRun = modelDetails["modelsToRun"]
        metricNames = modelDetails["metricNames"]
        if self.data:
            self.computeWidget = ComputeWidget(modelsToRun, metricNames, self.data)
            # DON'T WANT TO DISPLAY RESULTS IN ANOTHER WINDOW
            # WANT TO DISPLAY ON TAB 2/3
            self.computeWidget.results.connect(self.onEstimationComplete)     # signal emitted when estimation complete

    def onEstimationComplete(self, results):
        """
        description to be created at a later time
        """
        self.estimationComplete = True
        self.estimationResults = results
        self._main.tabs.tab1.sideMenu.runButton.setEnabled(True)    # re-enable button, can run another estimation
        # self.setDataView("view", self.dataViewIndex)
        self.updateUI()
        # set initial model selected
        # set plot
        print(results)

    def importFile(self):
        """
        Import selected file
        """
        self._main.tabs.tab1.sideMenu.sheetSelect.clear()   # clear sheet names from previous file
        self._main.tabs.tab1.sideMenu.sheetSelect.addItems(self.data.sheetNames)    # add sheet names from new file

        self.setDataView("view", self.dataViewIndex)
        # self.setMetricList()

    def changeSheet(self, index):
        """
        Change the current sheet displayed

        Args:
            index : index of the sheet
        """
        self.data.currentSheet = index
        self.setDataView("view", self.dataViewIndex)
        self._main.tabs.tab1.plotAndTable.figure.canvas.draw()
        self.setMetricList()

    def setMetricList(self):
        self._main.tabs.tab1.sideMenu.metricListWidget.clear()
        if self.dataLoaded:
            dataframe = self.data.getData()
            self._main.tabs.tab1.sideMenu.metricListWidget.addItems(dataframe.columns.values[2:-1])
            logging.info("{0} covariate metrics on this sheet: {1}".format(self.data.numCovariates,
                                                                    dataframe.columns.values[2:-1]))

    def setDataView(self, viewType, index):
        """
        Set the data to be displayed. 
        Called whenever a menu item is changed

        Args:
            viewType: string that determines view
            index: index of the dataview list
        """
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
        """
        Changes plot between MVF and intensity
        """
        self._main.tabs.tab1.plotAndTable.tableWidget.setModel(self.data.getDataModel())
        dataframe = self.data.getData()
        self.plotSettings.plotType = "step"

        if index == 0:
            # MVF
            self.createMVFPlot(dataframe)
        if index == 1:
            # Intensity
            self.createIntensityPlot(dataframe)

        # redraw figures
        self.ax2.legend()
        self._main.tabs.tab1.plotAndTable.figure.canvas.draw()
        self._main.tabs.tab2.plotAndTable.figure.canvas.draw()

    def createMVFPlot(self, dataframe):
        """
        called by setDataView
        """
        self.ax = self.plotSettings.generatePlot(self.ax, dataframe.iloc[:, 0], dataframe["Cumulative"], title="MVF", xLabel="time", yLabel="failures")
        if self.estimationComplete:
            self.ax2 = self.plotSettings.generatePlot(self.ax2, dataframe.iloc[:, 0], dataframe["Cumulative"], title="MVF", xLabel="time", yLabel="failures")
            self.plotSettings.plotType = "plot"
            # for model in self.estimationResults.values():
            #     # add line for model if selected
            #     if model.name in self.selectedModelNames:
            #         self.plotSettings.addLine(self.ax2, model.t, model.mvfList, model.name)



            for modelName in self.selectedModelNames:
                # add line for model if selected
                model = self.estimationResults[modelName]
                self.plotSettings.addLine(self.ax2, model.t, model.mvfList, model.name)

    def createIntensityPlot(self, dataframe):
        """
        called by setDataView
        """
        self.ax = self.plotSettings.generatePlot(self.ax, dataframe.iloc[:, 0], dataframe.iloc[:, 1], title="Intensity", xLabel="time", yLabel="failures")
        if self.estimationComplete:
            self.ax2 = self.plotSettings.generatePlot(self.ax2, dataframe.iloc[:, 0], dataframe.iloc[:, 1], title="Intensity", xLabel="time", yLabel="failures")
            self.plotSettings.plotType = "plot"
            # for model in self.estimationResults.values():
            #     # add line for model if selected
            #     if model.name in self.selectedModelNames:
            #         self.plotSettings.addLine(self.ax2, model.t, model.intensityList, model.name)

            for modelName in self.selectedModelNames:
                # add line for model if selected
                model = self.estimationResults[modelName]
                self.plotSettings.addLine(self.ax2, model.t, model.intensityList, model.name)

    def changePlot2(self, selectedModels):
        self.selectedModelNames = selectedModels
        self.updateUI()


    def setTrendTest(self, index):
        """
        description to be created at a later time
        """
        pass

    def setPlotStyle(self, style='-o', plotType="step"):
        """
        description to be created at a later time
        """
        self.plotSettings.style = style
        self.plotSettings.plotType = plotType
        self.updateUI()
        # self.setDataView("view", self.dataViewIndex)

    def updateUI(self):
        """
        Change Plot, Table and SideMenu
        when the state of the Data object changes

        Should be called explicitly
        """
        self.setDataView(self.viewType, self.dataViewIndex)

    def setupMenu(self):
        """
        description to be created at a later time
        """
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
    """
    description to be created at a later time
    """
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

#region Tab 1
class Tab1(QWidget):
    def __init__(self):
        super().__init__()
        self.setupTab1()

    def setupTab1(self):
        self.horizontalLayout = QHBoxLayout()       # main layout

        self.sideMenu = SideMenu1()
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

class SideMenu1(QVBoxLayout):
    """
    Side menu for tab 1
    """

    # signals
    viewChangedSignal = pyqtSignal(str, int)    # should this be before init?
    runModelSignal = pyqtSignal(dict)

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
        self.runButton.clicked.connect(self.emitRunModelsSignal)
        self.addWidget(self.runButton)

        self.addStretch(1)

        # signals
        self.sheetSelect.currentIndexChanged.connect(self.emitSheetChangedSignal)

    def setupSheetGroup(self):
        sheetGroupLayout = QVBoxLayout()
        # sheetGroupLayout.addWidget(QLabel("Select sheet"))

        self.sheetSelect = QComboBox()
        sheetGroupLayout.addWidget(self.sheetSelect)

        return sheetGroupLayout

    def setupModelsGroup(self):
        modelGroupLayout = QVBoxLayout()
        self.modelListWidget = QListWidget()

        # TEMPORARY
        # will later dynamically add model names
        loadedModels = [model.name for model in models.modelList.values()]
        self.modelListWidget.addItems(loadedModels)
        logging.info("{0} model(s) loaded: {1}".format(len(loadedModels), loadedModels))
        self.modelListWidget.setSelectionMode(QAbstractItemView.MultiSelection)       # able to select multiple models
        modelGroupLayout.addWidget(self.modelListWidget)

        return modelGroupLayout

    def setupMetricsGroup(self):
        metricsGroupLayout = QVBoxLayout()
        self.metricListWidget = QListWidget()   # metric names added dynamically from data when loaded
        self.metricListWidget.setSelectionMode(QAbstractItemView.MultiSelection)     # able to select multiple metrics
        metricsGroupLayout.addWidget(self.metricListWidget)

        return metricsGroupLayout

    def emitRunModelsSignal(self):
        """
        Method called when Run Estimation button pressed.
        Signal that tells models to run (runModelSignal) is
        only emitted if at least one model and at least one
        metric is selected.
        """
        logging.info("Run button pressed.")
        # get model names as strings
        
        selectedModelNames = [item.text() for item in self.modelListWidget.selectedItems()]
        # get model classes from models folder
        modelsToRun = [model for model in models.modelList.values() if model.name in selectedModelNames]
        # get selected metric names (IMPORTANT: returned in order they were clicked)
        selectedMetricNames = [item.text() for item in self.metricListWidget.selectedItems()]
        # sorts metric names in their order from the data file (left to right)
        metricNames = [self.metricListWidget.item(i).text() for i in range(self.metricListWidget.count()) if self.metricListWidget.item(i).text() in selectedMetricNames]
        # only emit the run signal if at least one model and at least one metric chosen
        if selectedModelNames and selectedMetricNames:
            self.runButton.setEnabled(False)    # disable button until estimation complete
            self.runModelSignal.emit({"modelsToRun": modelsToRun,
                                  "metricNames": metricNames})
            logging.info("Run models signal emitted. Models = {0}, metrics = {1}".format(selectedModelNames, selectedMetricNames))
        elif self.modelListWidget.count() > 0 and self.metricListWidget.count() > 0:
            # data loaded but not selected
            logging.warning("Must select at least one model and at least one metric.")
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setText("Model or metric not selected")
            msgBox.setInformativeText("Please select at least one model and at least one metric.")
            msgBox.setWindowTitle("Warning")
            msgBox.exec_()
        else:
            logging.warning("No data found. Data must be loaded in CSV or Excel format.")
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setText("No data found")
            msgBox.setInformativeText("Please load failure data as a .csv file or an Excel workbook (.xls, xlsx).")
            msgBox.setWindowTitle("Warning")
            msgBox.exec_()

    def emitSheetChangedSignal(self):
        self.viewChangedSignal.emit("sheet", self.sheetSelect.currentIndex())
#endregion

#region Tab 2
class Tab2(QWidget):
    def __init__(self):
        super().__init__()
        self.setupTab2()

    def setupTab2(self):
        self.horizontalLayout = QHBoxLayout()       # main layout

        '''
        self.plotGroup = QGroupBox("Model Results")
        self.plotGroup.setLayout(self.setupPlotGroup())
        self.horizontalLayout.addWidget(self.plotGroup, 60)

        self.tableGroup = QGroupBox("Table of Predictions")
        self.tableGroup.setLayout(self.setupTableGroup())
        self.horizontalLayout.addWidget(self.tableGroup, 40)
        '''

        self.sideMenu = SideMenu2()
        self.horizontalLayout.addLayout(self.sideMenu, 25)
        self.plotAndTable = PlotAndTable()
        self.horizontalLayout.addWidget(self.plotAndTable, 75)
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
        self.modelsGroup = QGroupBox("Select Model(s)")
        self.modelsGroup.setLayout(self.setupModelsGroup())
        self.addWidget(self.modelsGroup)

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

    def addSelectedModels(self, modelDetails):
        self.modelListWidget.clear()
        modelsRan = modelDetails["modelsToRun"]
        # metricNames = modelDetails["metricNames"]

        loadedModels = [model.name for model in modelsRan]
        self.modelListWidget.addItems(loadedModels)

    def emitModelChangedSignal(self):
        selectedModelNames = [item.text() for item in self.modelListWidget.selectedItems()]
        logging.info("Selected models: {0}".format(selectedModelNames))
        self.modelChangedSignal.emit(selectedModelNames)

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