#-----------------------------------------------------------------------------------#
# TODO:
# make sure everything that needs to be is a np array, not list
# MSE vs SSE?
# checking on other datasets
# making UI easier to use for someone who doesn't understand the math
# options selected from menubar like in SFRAT
# protection levels, access modifiers, public/private variables, properties
# fewer classes?
#   example: self._main.tabs.tab1.sideMenu.sheetSelect.addItems(self.data.sheetNames)
# dialog asking if you want to quit?
# pay attention to how scaling/strecting works, minimum sizes for UI elements
# use logging object, removes matplotlib debug messages in debug mode
# predict points? (commonWidgets)
# naming "hazard functions" instead of models
# fsolve doesn't return if converged, so it's not updated for models
#   should try other scipy functions
# self.viewType is never updated, we don't use updateUI()
# sometimes metric list doesn't load until interacted with
#------------------------------------------------------------------------------------#

# For handling debug output
import logging as log

# PyQt5 imports for UI elements
from PyQt5.QtWidgets import QMainWindow, qApp, QWidget, QTabWidget, QVBoxLayout, \
                            QAction, QActionGroup, QFileDialog
from PyQt5.QtCore import pyqtSignal

# Matplotlib imports for graphs/plots
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.figure import Figure
# import matplotlib.pyplot as plt

# numpy for fast array operations
# import numpy as np

# Local imports
import models
from ui.commonWidgets import ComputeWidget, SymbolicThread
from ui.tab1 import Tab1
from ui.tab2 import Tab2
from ui.tab3 import Tab3
from ui.tab4 import Tab4
from core.dataClass import Data
from core.graphSettings import PlotSettings
from core.allocation import EffortAllocation
from core.trendTests import *

class MainWindow(QMainWindow):
    """
    The main application window, called by running main.py
    """
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
        self.trendTests = {cls.__name__: cls for
                           cls in TrendTest.__subclasses__()}
        self.plotSettings = PlotSettings()
        self.selectedModelNames = []

        # self.estimationResults
        # self.currentModel

        # flags
        self.dataLoaded = False
        self.estimationComplete = False
        self.symbolicComplete = False

        # tab 1 plot and table
        self.ax = self._main.tabs.tab1.plotAndTable.figure.add_subplot(111)
        # tab 2 plot and table
        self.ax2 = self._main.tabs.tab2.plot.figure.add_subplot(111)

        # signal connections
        self.importFileSignal.connect(self.importFile)
        self._main.tabs.tab1.sideMenu.viewChangedSignal.connect(self.setDataView)
        self._main.tabs.tab1.sideMenu.runModelSignal.connect(self.runModels)    # run models when signal is received
        # self._main.tabs.tab1.sideMenu.runModelSignal.connect(self._main.tabs.tab2.sideMenu.addSelectedModels)    # fill tab 2 models group with selected models
        self._main.tabs.tab2.sideMenu.modelChangedSignal.connect(self.changePlot2)
        # connect tab2 list changed to refreshing tab 2 plot
        self._main.tabs.tab3.sideMenu.comboBoxChangedSignal.connect(self.runGoodnessOfFit)
        self._main.tabs.tab4.sideMenu.runAllocationSignal.connect(self.runAllocation)

        self.initUI()
        log.info("UI loaded.")

    def closeEvent(self, event):
        """
        Called when application is closed by user. Quits all threads,
        and shuts down app.
        """
        log.info("Covariate Tool application closed.")

        # --- stop running threads ---
        # stop symbolic thread 
        try:
            # self.symbolicThread.quit()
            self.symbolicThread.abort = True
            self.symbolicThread.wait()
        except AttributeError:
            pass

        # stop model estimation thread
        try:
            # self.computeWidget.computeTask.quit()
            self.computeWidget.computeTask.abort = True
            self.computeWidget.computeTask.wait()
        except AttributeError:
            # should catch if computeWidget not an attribute of mainWindow,
            # or if computeTask not yet an attribute of computeWidget
            pass
        
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
        self.mvf = QAction("MVF Graph", self, checkable=True)
        self.mvf.setShortcut("Ctrl+M")
        self.mvf.setStatusTip("Graphs display MVF of data")
        self.mvf.setChecked(True)
        self.mvf.triggered.connect(self.setMVFView)
        graphStyle.addAction(self.mvf)
        # intensity
        self.intensity = QAction("Intensity Graph", self, checkable=True)
        self.intensity.setShortcut("Ctrl+I")
        self.intensity.setStatusTip("Graphs display failure intensity")
        self.intensity.triggered.connect(self.setIntensityView)
        graphStyle.addAction(self.intensity)
        # trend test
        viewTest = QAction("View Trend", self, checkable=True)
        viewTest.setShortcut('Ctrl+T')
        viewTest.setStatusTip('View Trend Test')
        viewTest.triggered.connect(self._main.tabs.tab1.sideMenu.testChanged)
        graphStyle.addAction(viewTest)
        # add actions to view menu
        viewMenu.addSeparator()
        viewMenu.addActions(graphStyle.actions())

    #region Importing, plotting
    def fileOpened(self):
        """
        sets self.dataLoaded = True flag
        """
        # default location is datasets directory
        files = QFileDialog.getOpenFileName(self, "Open profile", "datasets", filter=("Data Files (*.csv *.xls *.xlsx)"))
        # if a file was loaded
        if files[0]:
            self._main.tabs.tab1.sideMenu.runButton.setDisabled(True)
            self.symbolicComplete = False   # reset flag, need to run symbolic functions before estimation
            self.data.importFile(files[0])      # imports loaded file
            self.dataLoaded = True
            log.info("Data loaded from %s", files[0])
            self.importFileSignal.emit()            # emits signal that file was imported successfully

            self.runSymbolic()

    def importFile(self):
        """
        Loads imported data into UI
        """
        self._main.tabs.tab1.sideMenu.sheetSelect.clear()   # clear sheet names from previous file
        self._main.tabs.tab1.sideMenu.sheetSelect.addItems(self.data.sheetNames)    # add sheet names from new file

        self.setDataView("view", self.dataViewIndex)
        # self.setMetricList()

    def runSymbolic(self):
        log.info("ENTERING runSymbolic FUNCTION")
        self.symbolicThread = SymbolicThread(models.modelList, self.data)
        self.symbolicThread.symbolicSignal.connect(self.onSymbolicComplete)
        self.symbolicThread.start()

        # MOVED TO commonWidgets, SymbolicThread class
        # log.info(f"modelList = {models.modelList}")
        # for m in models.modelList.values():
        #     # need to initialize models so they have the imported data
        #     instantiatedModel = m(data=self.data.getData(), metricNames=self.data.metricNames)
        #     m.lambdaFunctionAll = instantiatedModel.symAll()
        #     log.info(f"Lambda function created for {m.name} model")

    def onSymbolicComplete(self):
        log.info("ENTERING runSymbolic FUNCTION")
        self.symbolicComplete = True
        log.info("Symbolic calculations completed.")
        self._main.tabs.tab1.sideMenu.runButton.setDisabled(False)

    def changeSheet(self, index):
        """
        Change the current sheet displayed

        Args:
            index : index of the sheet
        """
        self.data.currentSheet = index      # store 
        self.setDataView("view", self.dataViewIndex)
        self._main.tabs.tab1.plotAndTable.figure.canvas.draw()
        self.setMetricList()

    def setMetricList(self):
        self._main.tabs.tab1.sideMenu.metricListWidget.clear()
        if self.dataLoaded:
            self._main.tabs.tab1.sideMenu.metricListWidget.addItems(self.data.metricNameCombinations)
            log.info("%d covariate metrics on this sheet: %s", self.data.numCovariates,
                                                               self.data.metricNames)

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
                self.dataViewIndex = index  # was at the end of elif statements, but would change mvf/intensity view
                                            # unintentionally when changing sheets
            elif viewType == "trend":
                self.setTrendTest(index)
            elif viewType == "sheet":
                self.changeSheet(index)
            #self.viewType = viewType
                # removed since it would change the sheet displayed when changing display settings

    def setRawDataView(self, index):
        """
        Changes plot between MVF and intensity
        """
        self._main.tabs.tab1.plotAndTable.tableWidget.setModel(self.data.getDataModel())
        dataframe = self.data.getData()
        self.plotSettings.plotType = "step"

        if self.dataViewIndex == 0:     # changed from index to self.dataViewIndex
            # MVF
            self.mvf.setChecked(True)
            self.createMVFPlot(dataframe)
        if self.dataViewIndex == 1:     # changed from index to self.dataViewIndex
            # Intensity
            self.intensity.setChecked(True)
            self.createIntensityPlot(dataframe)

        # redraw figures
        self.ax2.legend()
        self._main.tabs.tab1.plotAndTable.figure.canvas.draw()
        self._main.tabs.tab2.plot.figure.canvas.draw()

    def setTrendTest(self, index):
        """
        Set the view to a trend test

        Args:
            index: index of the list of trend test
        """
        trendTest = list(self.trendTests.values())[index]()
        trendData = trendTest.run(self.data.getData())
        self.ax = self.plotSettings.generatePlot(self.ax, trendData['X'],
                                                 trendData['Y'],
                                                 title=trendTest.name,
                                                 xLabel=trendTest.xAxisLabel,
                                                 yLabel=trendTest.yAxisLabel)
        self._main.tabs.tab1.plotAndTable.figure.canvas.draw()

    def createMVFPlot(self, dataframe):
        """
        called by setDataView
        """
        # self.plotSettings.plotType = "plot" # if continous
        self.plotSettings.plotType = "step" # if step

        self._main.tabs.tab1.sideMenu.testSelect.setDisabled(True)  # disable trend tests when displaying imported data
        self._main.tabs.tab1.sideMenu.confidenceSpinBox.setDisabled(True)

        self.ax = self.plotSettings.generatePlot(self.ax, dataframe['T'], dataframe["CFC"],
                                                 title="", xLabel="Cumulative time", yLabel="Cumulative failures")
        if self.estimationComplete:
            self.ax2 = self.plotSettings.generatePlot(self.ax2, dataframe['T'], dataframe["CFC"],
                                                      title="", xLabel="Cumulative time", yLabel="Cumulative failures")
            self.plotSettings.plotType = "plot"
            # for model in self.estimationResults.values():
            #     # add line for model if selected
            #     if model.name in self.selectedModelNames:
            #         self.plotSettings.addLine(self.ax2, model.t, model.mvfList, model.name)


            # model name and metric combination!
            for modelName in self.selectedModelNames:
                # add line for model if selected
                model = self.estimationResults[modelName]
                self.plotSettings.addLine(self.ax2, model.t, model.mvfList, modelName)

    def createIntensityPlot(self, dataframe):
        """
        called by setDataView
        """
        self.plotSettings.plotType = "bar"

        self._main.tabs.tab1.sideMenu.testSelect.setDisabled(True)  # disable trend tests when displaying imported data
        self._main.tabs.tab1.sideMenu.confidenceSpinBox.setDisabled(True)

        self.ax = self.plotSettings.generatePlot(self.ax, dataframe['T'], dataframe.iloc[:, 1],
                                                 title="", xLabel="Cumulative time", yLabel="Failures")
        if self.estimationComplete:
            self.ax2 = self.plotSettings.generatePlot(self.ax2, dataframe['T'], dataframe['FC'],
                                                      title="", xLabel="Cumulative time", yLabel="Failures")
            self.plotSettings.plotType = "plot"
            # for model in self.estimationResults.values():
            #     # add line for model if selected
            #     if model.name in self.selectedModelNames:
            #         self.plotSettings.addLine(self.ax2, model.t, model.intensityList, model.name)

            # model name and metric combination!
            for modelName in self.selectedModelNames:
                # add line for model if selected
                model = self.estimationResults[modelName]
                self.plotSettings.addLine(self.ax2, model.t, model.intensityList, modelName)

    def setPlotStyle(self, style='-o', plotType="step"):
        """
        description to be created at a later time
        """
        self.plotSettings.style = style
        self.plotSettings.plotType = plotType
        self.updateUI()
        # self.setDataView("view", self.dataViewIndex)

    def setLineView(self):
        self.setPlotStyle(style='-')
        log.info("Plot style set to line view.")

    def setPointsView(self):
        self.setPlotStyle(style='o', plotType='plot')
        log.info("Plot style set to points view.")

    def setLineAndPointsView(self):
        self.setPlotStyle(style='-o')
        log.info("Plot style set to line and points view.")

    def setMVFView(self):
        self.dataViewIndex = 0
        log.info("Data plots set to MVF view.")
        if self.dataLoaded:
            self.setRawDataView(self.dataViewIndex)

    def setIntensityView(self):
        self.dataViewIndex = 1
        log.info("Data plots set to intensity view.")
        if self.dataLoaded:
            self.setRawDataView(self.dataViewIndex)

    def changePlot2(self, selectedModels):
        self.selectedModelNames = selectedModels
        self.updateUI()

    def updateUI(self):
        """
        Change Plot, Table and SideMenu
        when the state of the Data object changes

        Should be called explicitly
        """
        self.setDataView(self.viewType, self.dataViewIndex)

    #endregion

    #region Estimation, allocation
    def runModels(self, modelDetails):
        """
        Run selected models using selected metrics

        Args:
            modelDetails : dictionary of models and metrics to use for calculations
        """
        # disable buttons until estimation complete
        self._main.tabs.tab1.sideMenu.runButton.setEnabled(False)
        self._main.tabs.tab4.sideMenu.allocationButton.setEnabled(False)
        modelsToRun = modelDetails["modelsToRun"]
        metricNames = modelDetails["metricNames"]
        if self.data:
            self.estimationComplete = False # estimation not complete since it just started running
            self._main.tabs.tab2.sideMenu.modelListWidget.clear()   # clear tab 2 list containing 
                                                                    # previously computed models,
                                                                    # only added when calculations complete
            self._main.tabs.tab4.sideMenu.modelListWidget.clear()
            self.computeWidget = ComputeWidget(modelsToRun, metricNames, self.data)
            # DON'T WANT TO DISPLAY RESULTS IN ANOTHER WINDOW
            # WANT TO DISPLAY ON TAB 2/3
            self.computeWidget.results.connect(self.onEstimationComplete)     # signal emitted when estimation complete

    def onEstimationComplete(self, results):
        """
        description to be created at a later time

        Args:
            results (dict): contains model objects
        """
        self.estimationComplete = True
        self.estimationResults = results
        self._main.tabs.tab1.sideMenu.runButton.setEnabled(True)    # re-enable button, can run another estimation
        self._main.tabs.tab4.sideMenu.allocationButton.setEnabled(True)      # re-enable allocation button, can't run
                                                                    # if estimation not complete
        # self.setDataView("view", self.dataViewIndex)
        self.updateUI()
        # set initial model selected
        # set plot

        convergedNames = []
        nonConvergedNames = []
        for key, model in results.items():
            if model.converged:
                convergedNames.append(key)
            else:
                nonConvergedNames.append(key)

        self._main.tabs.tab2.sideMenu.addSelectedModels(convergedNames) # add models to tab 2 list
                                                                        # so they can be selected
        self._main.tabs.tab2.sideMenu.addNonConvergedModels(nonConvergedNames)
                                                                        # show which models didn't converge
        self._main.tabs.tab3.addResultsToTable(results)
        self._main.tabs.tab4.sideMenu.addSelectedModels(convergedNames) # add models to tab 4 list so they
                                                                        # can be selected for allocation
        log.info("Estimation results: %s", results)

    def runGoodnessOfFit(self):
        if self.estimationComplete:
            # self._main.tabs.tab3.sideMenu.goodnessOfFit(self.estimationResults)
            self._main.tabs.tab3.addResultsToTable(self.estimationResults)

    def runAllocation(self, combinations):
        B = self._main.tabs.tab4.sideMenu.budgetSpinBox.value()     # budget
        f = self._main.tabs.tab4.sideMenu.failureSpinBox.value()    # number of failures (UNUSED)
        # m = self.estimationResults[combinations[0]]     # model object

        self.allocationResults = {}    # create a dictionary for allocation results
        for i in range(len(combinations)):
            name = combinations[i]
            if " - (No covariates)" not in name:
                m = self.estimationResults[name]  # model indexed by the name
                self.allocationResults[name] = [EffortAllocation(m, B, f), m]

        self._main.tabs.tab4.addResultsToTable(self.allocationResults, self.data)

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

        self.tab4 = Tab4()
        self.addTab(self.tab4, "Effort Allocation")

        self.resize(300, 200)