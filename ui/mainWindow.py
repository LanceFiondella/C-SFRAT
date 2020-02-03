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
# names of tabs in tab 2?
# self.viewType is never updated, we don't use updateUI()
# sometimes metric list doesn't load until interacted with
#------------------------------------------------------------------------------------#

# PyQt5 imports for UI elements
from PyQt5.QtWidgets import QMainWindow, qApp, QMessageBox, QWidget, QTabWidget, \
                            QHBoxLayout, QVBoxLayout, QTableView, QLabel, \
                            QLineEdit, QGroupBox, QComboBox, QListWidget, \
                            QPushButton, QAction, QActionGroup, QAbstractItemView, \
                            QFileDialog, QCheckBox, QScrollArea, QGridLayout, \
                            QTableWidget, QTableWidgetItem, QAbstractScrollArea, \
                            QSpinBox, QDoubleSpinBox
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QIntValidator, QDoubleValidator

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
from core.allocation import EffortAllocation
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
        self.ax2 = self._main.tabs.tab2.plot.figure.add_subplot(111)

        # signal connections
        self.importFileSignal.connect(self.importFile)
        self._main.tabs.tab1.sideMenu.viewChangedSignal.connect(self.setDataView)
        self._main.tabs.tab1.sideMenu.runModelSignal.connect(self.runModels)    # run models when signal is received
        # self._main.tabs.tab1.sideMenu.runModelSignal.connect(self._main.tabs.tab2.sideMenu.addSelectedModels)    # fill tab 2 models group with selected models
        self._main.tabs.tab2.sideMenu.modelChangedSignal.connect(self.changePlot2)
        # connect tab2 list changed to refreshing tab 2 plot
        self._main.tabs.tab4.sideMenu.runAllocationSignal.connect(self.runAllocation)

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
        logging.info("Estimation results: {0}".format(results))

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
        self.data.currentSheet = index      # store 
        self.setDataView("view", self.dataViewIndex)
        self._main.tabs.tab1.plotAndTable.figure.canvas.draw()
        self.setMetricList()

    def setMetricList(self):
        self._main.tabs.tab1.sideMenu.metricListWidget.clear()
        if self.dataLoaded:
            self._main.tabs.tab1.sideMenu.metricListWidget.addItems(self.data.metricNameCombinations)
            logging.info("{0} covariate metrics on this sheet: {1}".format(self.data.numCovariates,
                                                                    self.data.metricNames))

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
            self.createMVFPlot(dataframe)
        if self.dataViewIndex == 1:     # changed from index to self.dataViewIndex
            # Intensity
            self.createIntensityPlot(dataframe)

        # redraw figures
        self.ax2.legend()
        self._main.tabs.tab1.plotAndTable.figure.canvas.draw()
        self._main.tabs.tab2.plot.figure.canvas.draw()

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


            # model name and metric combination!
            for modelName in self.selectedModelNames:
                # add line for model if selected
                model = self.estimationResults[modelName]
                self.plotSettings.addLine(self.ax2, model.t, model.mvfList, modelName)

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

            # model name and metric combination!
            for modelName in self.selectedModelNames:
                # add line for model if selected
                model = self.estimationResults[modelName]
                self.plotSettings.addLine(self.ax2, model.t, model.intensityList, modelName)

    def changePlot2(self, selectedModels):
        self.selectedModelNames = selectedModels
        self.updateUI()

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

        # # cons = ({'type': 'ineq', 'fun': lambda x:  B-x[0]-x[1]-x[2]})
        # cons = ({'type': 'ineq', 'fun': lambda x:  B - sum([x[i] for i in range(m.numCovariates)])})
        # # bnds = ((0, None), (0, None), (0, None))
        # bnds = tuple((0, None) for i in range(m.numCovariates))

        # res = shgo(m.allocationFunction, args=(f,), bounds=bnds, constraints=cons)#, n=10000, iters=4)
        # # res = shgo(lambda x: -(51+ 1.5449911694401008*(1- (0.9441308828628996 ** (np.exp(0.10847739229960603*x[0]+0.027716725008716442*x[1]+0.159319065848297*x[2]))))), bounds=bnds, constraints=cons, n=10000, iters=4)
        # print(res)
        # print(sum(res.x))

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

        self.tab4 = Tab4()
        self.addTab(self.tab4, "Effort Allocation")

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
        self.plotAndTable = PlotAndTable("Plot", "Table")
        self.horizontalLayout.addWidget(self.plotAndTable, 75)

        self.setLayout(self.horizontalLayout)

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

        buttonLayout = QHBoxLayout()
        self.selectAllButton = QPushButton("Select All")
        self.clearAllButton = QPushButton("Clear All")
        self.selectAllButton.clicked.connect(self.selectAll)
        self.clearAllButton.clicked.connect(self.clearAll)
        buttonLayout.addWidget(self.selectAllButton, 50)
        buttonLayout.addWidget(self.clearAllButton, 50)
        metricsGroupLayout.addLayout(buttonLayout)

        return metricsGroupLayout
    
    def selectAll(self):
        self.metricListWidget.selectAll()
        self.metricListWidget.repaint()

    def clearAll(self):
        self.metricListWidget.clearSelection()
        self.metricListWidget.repaint()

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
        selectedMetricNames = [item.text().split(", ") for item in self.metricListWidget.selectedItems()]
            # split combinations
        # sorts metric names in their order from the data file (left to right)
        #metricNames = [self.metricListWidget.item(i).text() for i in range(self.metricListWidget.count()) if self.metricListWidget.item(i).text() in selectedMetricNames]
        # only emit the run signal if at least one model and at least one metric chosen
        if selectedModelNames and selectedMetricNames:
            # self.runButton.setEnabled(False)    # disable button until estimation complete
            self.runModelSignal.emit({"modelsToRun": modelsToRun,
                                      "metricNames": selectedMetricNames})
                                      #"metricNames": metricNames})
                                      
            logging.info("Run models signal emitted. Models = {0}, metrics = {1}".format(selectedModelNames, selectedMetricNames))
        elif self.modelListWidget.count() > 0 and self.metricListWidget.count() > 0:
            # data loaded but not selected
            logging.warning("Must select at least one model.")
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setText("Model not selected")
            msgBox.setInformativeText("Please select at least one model and at least one metric option.")
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
        self.sideMenu = SideMenu2()
        self.horizontalLayout.addLayout(self.sideMenu, 25)
        self.plot = PlotWidget()
        self.horizontalLayout.addWidget(self.plot, 75)
        self.setLayout(self.horizontalLayout)

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
        self.modelsGroup = QGroupBox("Select Model Results")
        self.nonConvergedGroup = QGroupBox("Did Not Converge")
        self.modelsGroup.setLayout(self.setupModelsGroup())
        self.nonConvergedGroup.setLayout(self.setupNonConvergedGroup())
        self.addWidget(self.modelsGroup, 60)
        self.addWidget(self.nonConvergedGroup, 40)

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

    def setupNonConvergedGroup(self):
        nonConvergedGroupLayout = QVBoxLayout()
        self.nonConvergedListWidget = QListWidget()
        nonConvergedGroupLayout.addWidget(self.nonConvergedListWidget)

        return nonConvergedGroupLayout

    def addSelectedModels(self, modelNames):
        """


        Args:
            modelNames (list): list of strings, name of each model
        """

        #self.modelListWidget.clear()
        # modelsRan = modelDetails["modelsToRun"]
        # metricNames = modelDetails["metricNames"]

        # loadedModels = [model.name for model in modelsRan]
        # self.modelListWidget.addItems(loadedModels)

        self.modelListWidget.addItems(modelNames)

    def addNonConvergedModels(self, nonConvergedNames):
        self.nonConvergedListWidget.addItems(nonConvergedNames)

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
        self.setupTable()
        self.horizontalLayout.addWidget(self.table)
        self.setLayout(self.horizontalLayout)

    def setupTable(self):
        self.table = QTableWidget()
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)     # make cells unable to be edited
        self.table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
                                                                    # column width fit to contents
        self.table.setRowCount(1)
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(["Model Name", "Covariates", "Log-Likelihood", "AIC", "BIC", "SSE", "AHP"])
        self.table.move(0,0)

    def addResultsToTable(self, results):
        self.table.setSortingEnabled(False) # disable sorting while editing contents
        self.table.clear()
        self.table.setHorizontalHeaderLabels(["Model Name", "Covariates", "Log-Likelihood", "AIC", "BIC", "SSE", "AHP"])
        self.table.setRowCount(len(results))    # set row count to include all model results, 
                                                # even if not converged
        i = 0   # number of converged models
        for key, model in results.items():
            if model.converged:
                self.table.setItem(i, 0, QTableWidgetItem(model.name))
                self.table.setItem(i, 1, QTableWidgetItem(model.metricString))
                self.table.setItem(i, 2, QTableWidgetItem("{0:.2f}".format(model.llfVal)))
                self.table.setItem(i, 3, QTableWidgetItem("{0:.2f}".format(model.aicVal)))
                self.table.setItem(i, 4, QTableWidgetItem("{0:.2f}".format(model.bicVal)))
                self.table.setItem(i, 5, QTableWidgetItem("{0:.2f}".format(model.sseVal)))
                i += 1
        self.table.setRowCount(i)   # set row count to only include converged models
        self.table.resizeColumnsToContents()    # resize column width after table is edited
        self.table.setSortingEnabled(True)      # re-enable sorting after table is edited
#endregion

#region tab4_test
class Tab4(QWidget):

    def __init__(self):
        super().__init__()
        self.setupTab4()

    def setupTab4(self):
        self.mainLayout = QHBoxLayout() # main tab layout

        self.sideMenu = SideMenu4()
        self.mainLayout.addLayout(self.sideMenu, 25)
        self.table = self.setupTable()
        self.mainLayout.addWidget(self.table, 75)
        self.setLayout(self.mainLayout)

    def setupTable(self):
        table = QTableWidget()
        table.setEditTriggers(QTableWidget.NoEditTriggers)     # make cells unable to be edited
        table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
                                                                    # column width fit to contents
        table.setRowCount(1)
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Model Name", "Covariates", "H"])
        table.move(0,0)

        return table

    def createHeaderLabels(self, metricNames):
        percentNames = []
        i = 0
        for name in metricNames:
            percentNames.append("%" + name)
            i += 1
        headerLabels = ["Model Name", "Covariates", "H"] + percentNames
        return headerLabels

    def addResultsToTable(self, results, data):
        """
        results = dict
            results[name] = [EffortAllocation, Model]
        """
        self.table.setSortingEnabled(False) # disable sorting while editing contents
        self.table.clear()
        self.table.setColumnCount(3 + len(data.metricNames))
        self.table.setHorizontalHeaderLabels(self.createHeaderLabels(data.metricNames))
        self.table.setRowCount(len(results))    # set row count to include all model results, 
                                                # even if not converged
        i = 0   # rows

        for key, value in results.items():
            res = value[0]
            model = value[1]

            print(res.percentages)

            self.table.setItem(i, 0, QTableWidgetItem(model.name))   # model name
            self.table.setItem(i, 1, QTableWidgetItem(model.metricString))  # model metrics
            self.table.setItem(i, 2, QTableWidgetItem("{0:.2f}".format(res.H)))
            # number of columns = number of covariates
            j = 0
            for name in model.metricNames:
                col = data.metricNameDictionary[name]
                self.table.setItem(i, 3 + col, QTableWidgetItem("{0:.2f}".format(res.percentages[j])))
                j += 1
            i += 1

                # try:
                #     c = model.metricNameDictionary[model.metricNames[j]]    # get index from metric name
                #     self.table.setItem(i, 2 + j, QTableWidgetItem(str(c)))  # 
                # except KeyError:
                #     self.table.setItem(i, )
        self.table.setRowCount(i)   # set row count to only include converged models
        self.table.resizeColumnsToContents()    # resize column width after table is edited
        self.table.setSortingEnabled(True)      # re-enable sorting after table is edited

class SideMenu4(QVBoxLayout):
    """
    Side menu for tab 4
    """

    # signals
    runAllocationSignal = pyqtSignal(list)  # starts allocation computation

    def __init__(self):
        super().__init__()
        self.setupSideMenu()

    def setupSideMenu(self):
        self.modelsGroup = QGroupBox("Select Models/Metrics for Allocation")
        self.modelsGroup.setLayout(self.setupModelsGroup())
        self.optionsGroup = QGroupBox("Allocation Parameters")
        self.optionsGroup.setLayout(self.setupOptionsGroup())
        self.setupAllocationButton()

        self.addWidget(self.modelsGroup, 75)
        self.addWidget(self.optionsGroup, 25)
        self.addWidget(self.allocationButton)

        self.addStretch(1)

    def setupModelsGroup(self):
        modelGroupLayout = QVBoxLayout()
        self.modelListWidget = QListWidget()
        modelGroupLayout.addWidget(self.modelListWidget)
        self.modelListWidget.setSelectionMode(QAbstractItemView.MultiSelection)       # able to select multiple models

        return modelGroupLayout

    def setupOptionsGroup(self):
        optionsGroupLayout = QVBoxLayout()
        optionsGroupLayout.addWidget(QLabel("Budget"))
        self.budgetSpinBox = QDoubleSpinBox()
        # self.budgetSpinBox.setMaximumWidth(200)
        self.budgetSpinBox.setRange(0.0, 999999.0)
        self.budgetSpinBox.setValue(20)
        optionsGroupLayout.addWidget(self.budgetSpinBox)
        
        optionsGroupLayout.addWidget(QLabel("Failures"))
        self.failureSpinBox = QSpinBox()
        # self.failureSpinBox.setMaximumWidth(200)
        self.failureSpinBox.setRange(1, 999999)
        optionsGroupLayout.addWidget(self.failureSpinBox)

        return optionsGroupLayout

    def setupAllocationButton(self):
        self.allocationButton = QPushButton("Run Allocation")
        self.allocationButton.setEnabled(False) # begins disabled since no model has been run yet
        # self.allocationButton.setMaximumWidth(250)
        self.allocationButton.clicked.connect(self.emitRunAllocationSignal)

    def addSelectedModels(self, modelNames):
        """


        Args:
            modelNames (list): list of strings, name of each model
        """

        self.modelListWidget.addItems(modelNames)

    def emitRunAllocationSignal(self):
        selectedCombinationNames = [item.text() for item in self.modelListWidget.selectedItems()]
        if selectedCombinationNames:
            selectedCombinationNames = [item.text() for item in self.modelListWidget.selectedItems()]
            logging.info("Selected for Allocation: {0}".format(selectedCombinationNames))
            self.runAllocationSignal.emit(selectedCombinationNames)
        else:
            logging.warning("Must select at least one model/metric combination for allocation.")
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setText("No selection made for allocation")
            msgBox.setInformativeText("Please select at least one model/metric combination.")
            msgBox.setWindowTitle("Warning")
            msgBox.exec_()

#endregion