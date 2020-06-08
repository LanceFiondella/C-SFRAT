"""Contains all UI elements, provides signal connections.

Contains highest level UI elements. Connects all core modules and functions to
the UI elements. Able to reference all elements and the signals they emit.

"""

###############################################################################
# TODO:
# make sure everything that needs to be is a np array, not list
# MSE vs SSE?
# checking on other datasets
# making UI easier to use for someone who doesn't understand the math
# options selected from menubar like in SFRAT
# protection levels, access modifiers, public/private variables, properties
# fewer classes?
#   example: self._main.tab1.sideMenu.sheetSelect.addItems(self.data.sheetNames)
# dialog asking if you want to quit?
# pay attention to how scaling/strecting works, minimum sizes for UI elements
# use logging object, removes matplotlib debug messages in debug mode
# naming "hazard functions" instead of models
# fsolve doesn't return if converged, so it's not updated for models
#   should try other scipy functions
# self.viewType is never updated
# sometimes metric list doesn't load until interacted with
# bar chart isn't ideal for large datasets
# clean up prediction plot
###############################################################################

# For handling debug output
import logging as log

# PyQt5 imports for UI elements
from PyQt5.QtWidgets import QMainWindow, qApp, QWidget, QTabWidget, \
                            QVBoxLayout, QAction, QActionGroup, QFileDialog
from PyQt5.QtCore import pyqtSignal

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
    """Window that is displayed when starting application.

    Provides top level control of application. Connects model functions and
    UI elements through signal connections. Handles file opening, running
    estimation/allocation/trend tests, creating/updating plots, menu options.

    Attributes:
        _main: Instance of MainWidget class, contains widgets.
        debug: Boolean indicating if debug mode is active or not.
        data: Pandas dataframe containing imported data.
        trendTests: A dict of trend test classes, indexed by class name as
            string.
        plotSettings: Instance of PlotSettings class, handles plotting.
        selectedModelNames: A list of selected model/metric combinations in
            tab 2 list widget.
        dataLoaded: Boolean flag indicating if data has been fully loaded from
            .xlsx/.csv file.
        estimationComplete: Boolean flag indicating if estimation has been
            completed (estimation started by selecting models/metrics on tab 1
            and clicking run estimation button).
        estimationResults: A dict containing instances of the model classes
            (one for each model/metric combination) selected for estimation.
            The dict is indexed by the name of the model/metric combination
            as a string. The variable is set after estimation is complete.
        symbolicComplete: Boolean flag indicating if the symbolic computation
            has been completed. Symbolic computations begin when a data file
            is loaded.
        ax: A matplotlib axes object, handles tab 1 plot.
        ax2: A matplotlib axes object, handles tab 2 plot.
        importFileSignal: Signal that is emitted when a file containing data
            is opened. Connects to importFile method that performs import.
        dataViewIndex: An int that stores which plot view is displayed. 0 is
            for MVF view, 1 is for intensity view.
        symbolicThread: SymbolicThread object (inherits from QThread) that runs
            symbolic calculations on separate thread. Stored as attribute to
            safely abort thread if application is closed before thread
            completes.
        computeWidget: ComputeWidget object containing model estimation thread.
            Stored as attribute to safely abort thread if application is closed
            before thread completes.
        menu: QMenuBar object containing all menu bar actions.
        mvf: QAction object controlling MVF view. Stored as attribute so it can
            be automatically checked if MVF view is set in a way that does not
            involve clicking the menu bar option.
        intensity: QAction object controlling intensity view. Stored as
            attribute so it can be automatically checked if intensity view is
            set in a way that does not involve clicking the menu bar option.
        allocationResults: A dict containing the results of the effort
            allocation, indexed by the name of the model/metric combination
            as a string.
    """

    # signals
    importFileSignal = pyqtSignal()

    def __init__(self, debug=False):
        """Inits MainWindow, not in debug mode by default."""
        super().__init__()

        self._main = MainWidget()
        self.setCentralWidget(self._main)

        # set debug mode
        self.debug = debug

        # set data
        self.data = Data()
        self.trendTests = {cls.__name__: cls for
                           cls in TrendTest.__subclasses__()}
        self.plotSettings = PlotSettings()
        self.selectedModelNames = []

        # self.estimationResults

        # flags
        self.dataLoaded = False
        self.estimationComplete = False
        self.symbolicComplete = False

        # tab 1 plot and table
        self.ax = self._main.tab1.plotAndTable.figure.add_subplot(111)
        # tab 2 plot and table
        self.ax2 = self._main.tab2.plot.figure.add_subplot(111)

        # SIGNAL CONNECTIONS
        self.importFileSignal.connect(self.importFile)
        self._main.tab1.sideMenu.viewChangedSignal.connect(self.setDataView)
        # run models when signal is received
        self._main.tab1.sideMenu.runModelSignal.connect(self.runModels)
        self._main.tab1.sideMenu.confidenceSignal.connect(self.updateLaplaceConfidencePlot)
        self._main.tab2.sideMenu.modelChangedSignal.connect(self.changePlot2)

        # connect tab2 list changed to refreshing tab 2 plot
        self._main.tab2.sideMenu.failureChangedSignal.connect(self.runPrediction)
        self._main.tab3.sideMenu.spinBoxChangedSignal.connect(self.runGoodnessOfFit)
        self._main.tab4.sideMenu.runAllocationSignal.connect(self.runAllocation)

        self._initUI()
        log.info("UI loaded.")

    def _initUI(self):
        """Sets window parameters, fonts, initializes UI elements."""
        # setup main window parameters
        title = "Covariate Tool"
        left = 100
        top = 100
        width = 1280
        height = 960
        minWidth = 1000
        minHeight = 800

        self._setupMenu()
        self.setWindowTitle(title)
        self.setGeometry(left, top, width, height)
        self.setMinimumSize(minWidth, minHeight)
        self.statusBar().showMessage("")
        # self.viewType = "view"
        self.dataViewIndex = 0

        # setup font
        # self.setStyleSheet("QLabel {font: 12pt Segoe}")
        # self.setStyleSheet("QGroupBox {font: 12pt Segoe}")
        # self.setStyleSheet("QTabWidget {font: 12pt Segoe}")
        # self.setStyleSheet("QListItem {font: 12pt Segoe}")
        # self.setStyleSheet("QPushButton {font: 12pt Segoe}")
        self.setStyleSheet("QWidget {font: 12pt Segoe}")

        self.show()

    def _setupMenu(self):
        """Initializes menu bar and menu actions.

        Menu bar contains two menus: File and View. File menu contains Open
        (opens file dialog for importing data file) and Exit (closes
        application) actions. View menu contains 3 groups: one for line style
        actions (points/lines), one for line type of the fitted data (step vs.
        smooth curve), and one for plot type (MVF, intensity, or trend test on
        tab 1).
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
        # submenu
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

        # -- line style (step vs smooth)
        lineStyle = QActionGroup(viewMenu)
        # step
        step = QAction("Step Plot", self, checkable=True)
        step.setShortcut("Ctrl+D")
        step.setStatusTip("Step curve for MVF lines")
        step.setChecked(True)
        step.triggered.connect(self.setStepLine)
        lineStyle.addAction(step)
        # smooth
        smooth = QAction("Smooth Plot", self, checkable=True)
        smooth.setShortcut("Ctrl+F")
        smooth.setStatusTip("Step curve for MVF lines")
        smooth.triggered.connect(self.setSmoothLine)
        lineStyle.addAction(smooth)
        # add actions to view menu
        viewMenu.addSeparator()
        viewMenu.addActions(lineStyle.actions())

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
        viewTest.triggered.connect(self._main.tab1.sideMenu.testChanged)
        graphStyle.addAction(viewTest)
        # add actions to view menu
        viewMenu.addSeparator()
        viewMenu.addActions(graphStyle.actions())

    def closeEvent(self, event):
        """Quits all threads, and shuts down app.

        Called when application is closed by user. Waits to abort symbolic and
        estimation threads safely if they are still running when application
        is closed.
        """
        log.info("Covariate Tool application closed.")

        # --- stop running threads ---
        # stop symbolic thread
        try:
            # self.symbolicThread.quit()
            self.symbolicThread.abort = True
            self.symbolicThread.wait()
        except (AttributeError, RuntimeError):
            # should do something with RuntimeError
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

    #region Importing, plotting
    def fileOpened(self):
        """Opens file dialog; sets flags and emits signals if file loaded.

        Action is only taken if a file is selected and opened using the file
        dialog. The importFile method is run, and the dataLoaded flag is set to
        True afterwards.The run estimation button on tab 1 is disabled, later
        enabled in a separate function when symbolic calculations are complete.
        The symbolicComplete flag is set to false before running the symbolic
        calculations.
        """
        # default location is datasets directory
        files = QFileDialog.getOpenFileName(self, "Open profile", "datasets",
                                            filter=("Data Files (*.csv *.xls *.xlsx)"))
        # if a file was loaded
        if files[0]:
            self._main.tab1.sideMenu.runButton.setDisabled(True)
            self.symbolicComplete = False   # reset flag, need to run symbolic
                                            # functions before estimation
            self.data.importFile(files[0])  # imports loaded file
            self.dataLoaded = True
            log.info("Data loaded from %s", files[0])
            self.importFileSignal.emit()    # emits signal that file was
                                            # imported successfully

            self.runSymbolic()

    def importFile(self):
        """Sets UI elements with imported data.

        Updates sheet select on tab 1 with sheet names (if applicable). Calls
        setDataView method to update tab 1 plot and table.
        """
        # clear sheet names from previous file
        self._main.tab1.sideMenu.sheetSelect.clear()
        # add sheet names from new file
        self._main.tab1.sideMenu.sheetSelect.addItems(self.data.sheetNames)

        self.setDataView("view", self.dataViewIndex)
        # self.setMetricList()

    def runSymbolic(self):
        """Initiates symbolic calculations that run on SymbolicThread.

        Called when data imported. Symbolic calculations performed for all
        models. Creates lambda function for LLF for combination of all
        covariates.
        """
        log.info("ENTERING runSymbolic FUNCTION")
        self.symbolicThread = SymbolicThread(models.modelList, self.data)
        self.symbolicThread.symbolicSignal.connect(self.onSymbolicComplete)
        self.symbolicThread.start()

    def onSymbolicComplete(self):
        """Sets symbolicComplete flag, emables tab 1 run estimation button."""
        log.info("ENTERING runSymbolic FUNCTION")
        self.symbolicComplete = True
        log.info("Symbolic calculations completed.")
        self._main.tab1.sideMenu.runButton.setEnabled(True)

    def redrawPlot(self, tabNumber):
        """Redraws plot for the provided tab number.

        Args:
            tabNumber: Tab number (int) that contains the figure to redraw.
        """
        if tabNumber == 1:
            self._main.tab1.plotAndTable.figure.canvas.draw()
        elif tabNumber == 2:
            # rescale plot: https://stackoverflow.com/questions/10944621/dynamically-updating-plot-in-matplotlib
            self.ax2.relim()
            self.ax2.autoscale_view()
            self._main.tab2.plot.figure.canvas.draw()

    def changeSheet(self, index):
        """Changes the current sheet displayed.

        Args:
            index: The index of the sheet (int).
        """
        self.data.currentSheet = index      # store
        self.setDataView("view", self.dataViewIndex)
        self.redrawPlot(1)
        self.setMetricList()

    def setMetricList(self):
        """Updates tab 1 list widget with metric names on current sheet."""
        self._main.tab1.sideMenu.metricListWidget.clear()
        if self.dataLoaded:
            # data class stores all combinations of metric names
            self._main.tab1.sideMenu.metricListWidget.addItems(self.data.metricNameCombinations)
            log.info("%d covariate metrics on this sheet: %s", self.data.numCovariates,
                                                               self.data.metricNames)

    def setDataView(self, viewType, index):
        """Sets the data to be displayed.

        Called whenever a menu item is changed, or when trend test changed.
        Three options for viewType: "view", "trend", or "sheet". The index
        controls which option of the selected viewType is selected.

        Args:
            viewType: String that determines if plot type, trend test, or sheet
                is set.
            index: Index (int) that determines which plot type, trend test, or
                sheet to display. Dependent on viewType.
        """
        # enable/disable confidence level spin box
        if self.data.getData() is not None:
            if viewType == "view":
                self.setRawDataView(index)
                self.dataViewIndex = index
            elif viewType == "trend":
                self.setTrendTest(index)
            elif viewType == "sheet":
                self.changeSheet(index)
            # self.viewType = viewType
                # removed since it would change the sheet displayed when
                # changing display settings

    def setRawDataView(self, index):
        """Creates MVF or intensity plot, based on index.

        Args:
            index: Integer that controls which plot to create. 0 creates MVF
                plot, 1 creates intensity plot.
        """
        self._main.tab1.plotAndTable.tableWidget.setModel(self.data.getDataModel())
        dataframe = self.data.getData()
        # self.plotSettings.plotType = "step"

        if self.dataViewIndex == 0:
            # MVF
            self.mvf.setChecked(True)
            self.createMVFPlot(dataframe)
        if self.dataViewIndex == 1:
            # Intensity
            self.intensity.setChecked(True)
            self.createIntensityPlot(dataframe)

        # redraw figures
        self.ax2.legend()
        self.redrawPlot(1)
        self.redrawPlot(2)

    def setTrendTest(self, index):
        """Sets the tab 1 plot to specified trend test.

        Args:
            index: Which trend test to generate a plot for, where 0 is Laplace
                and 1 is a running arithmetic average.
        """
        trendTest = list(self.trendTests.values())[index]()
        trendData = trendTest.run(self.data.getData())
        self.plotSettings.plotType = "step"  # want step plot for trend tests
        self.ax = self.plotSettings.generatePlot(self.ax, trendData['X'],
                                                 trendData['Y'],
                                                 title=trendTest.name,
                                                 xLabel=trendTest.xAxisLabel,
                                                 yLabel=trendTest.yAxisLabel)
        # add additional horizontal lines for confidence levels
        if self.dataLoaded and trendTest.name == "Laplace Trend Test":
            # enable spin box
            self._main.tab1.sideMenu.confidenceSpinBox.setEnabled(True)

            # add dotted lines, these don't change
            PlotSettings.addLaplaceLines(self.ax, self._main.tab1.sideMenu.confidenceSpinBox.value())

            # add line indicating user-specified confidence level
            # when Laplace plot first shown, use the current value of spinbox
            # PlotSettings.addSpecifiedConfidenceLine(self.ax, self._main.tab1.sideMenu.confidenceSpinBox.value())

        elif trendTest.name == "Running Arithmetic Average":
            self._main.tab1.sideMenu.confidenceSpinBox.setDisabled(True)
                                    
        self.redrawPlot(1)  # need to re-draw figure

    def updateLaplaceConfidencePlot(self, confidence):
        """Updates confidence line on Laplace trend test plot in tab 1.

        Args:
            confidence: Confidence level of the Laplace trend test, determines
                where horizontal line is drawn.
        """
        if self.dataLoaded:
            # update line indicating user-specified confidence level
            PlotSettings.updateConfidenceLine(self.ax, confidence)
            self.redrawPlot(1)  # need to re-draw figure

    def createMVFPlot(self, dataframe):
        """Creates MVF plots for tabs 1 and 2.

        Creates step plot for imported data. Tab 2 plot only displayed if
        estimation is complete. For fitted data, creates either a step or
        smooth plot, depending on what has been specified by the user in the
        menu bar. Called by setRawDataView method.
        """
        # self.plotSettings.plotType = "plot" # if continous
        # self.plotSettings.plotType = "step" # if step

        # save previous plot type, always want observed data to be step plot
        previousPlotType = self.plotSettings.plotType

        # disable trend tests when displaying imported data
        self._main.tab1.sideMenu.testSelect.setDisabled(True)
        self._main.tab1.sideMenu.confidenceSpinBox.setDisabled(True)

        # tab 1 plot
        self.plotSettings.plotType = "step"
        self.ax = self.plotSettings.generatePlot(self.ax, dataframe['T'], dataframe["CFC"],
                                                 title="", xLabel="Cumulative time", yLabel="Cumulative failures")

        # tab 2 plot
        if self.estimationComplete:
            self.ax2 = self.plotSettings.generatePlot(self.ax2, dataframe['T'], dataframe["CFC"],
                                                      title="", xLabel="Cumulative time", yLabel="Cumulative failures")

            self.plotSettings.plotType = previousPlotType   # want model fits to be plot type specified by user

            # add vertical line at last element of original data
            self.ax2.axvline(x=dataframe['T'].iloc[-1], color='red', linestyle='dotted')

            # self.plotSettings.plotType = "step"
            # model name and metric combination
            for modelName in self.selectedModelNames:
                # add line for model if selected
                model = self.estimationResults[modelName]
                self.plotSettings.addLine(self.ax2, model.t, model.mvfList, modelName)

    def createIntensityPlot(self, dataframe):
        """Creates intensity plots for tabs 1 and 2.

        Creates step plot for imported data. Tab 2 plot only displayed if
        estimation is complete. For fitted data, creates either a step or
        smooth plot, depending on what has been specified by the user in the
        menu bar. Called by setRawDataView method.
        """
        # need to change plot type to "bar" for intensity view, but want model result lines
        # to use whatever plot type had been selected
        # save the previous plot type, use it after bar plot created

        previousPlotType = self.plotSettings.plotType
        self.plotSettings.plotType = "bar"
        # self.plotSettings.plotType = "step"

        # disable trend tests when displaying imported data
        self._main.tab1.sideMenu.testSelect.setDisabled(True)
        self._main.tab1.sideMenu.confidenceSpinBox.setDisabled(True)

        self.ax = self.plotSettings.generatePlot(self.ax, dataframe['T'], dataframe.iloc[:, 1],
                                                 title="", xLabel="Cumulative time", yLabel="Failures")
        if self.estimationComplete:
            self.ax2 = self.plotSettings.generatePlot(self.ax2, dataframe['T'], dataframe['FC'],
                                                      title="", xLabel="Cumulative time", yLabel="Failures")
            self.plotSettings.plotType = previousPlotType

            # model name and metric combination!
            for modelName in self.selectedModelNames:
                # add line for model if selected
                model = self.estimationResults[modelName]
                self.plotSettings.addLine(self.ax2, model.t, model.intensityList, modelName)

    #region plot styles
    def setPlotStyle(self, style='-o'):
        """Updates plots with specified line style.

        Args:
            style: Matplotlib line style (string). Options included are line
                ('-'), points ('o'), and line and points ('-o').
        """
        self.plotSettings.style = style
        self.updateUI()

    def setLineView(self):
        """Sets plot style to line."""
        self.setPlotStyle(style='-')
        log.info("Plot style set to line view.")

    def setPointsView(self):
        """Sets plot style to points."""
        self.setPlotStyle(style='o')
        log.info("Plot style set to points view.")

    def setLineAndPointsView(self):
        """Sets plot style to line and points."""
        self.setPlotStyle(style='-o')
        log.info("Plot style set to line and points view.")
    #endregion

    #region plot types
    def setPlotType(self, plotType="step"):
        """Updates plot with specified plot type.

        Args:
            plotType: Matplotlib plot type (string). Options include 'step' and
                'plot' (smooth curve).
        """
        self.plotSettings.plotType = plotType
        self.updateUI()
        # self.setDataView("view", self.dataViewIndex)

    def setStepLine(self):
        """Sets plot type to step plot."""
        self.setPlotType(plotType="step")
        log.info("Line style set to 'step'.")

    def setSmoothLine(self):
        """Sets plot type to smooth line ('plot')"""
        self.setPlotType(plotType="plot")
        log.info("Line style set to 'smooth'.")
    #endregion

    def setMVFView(self):
        """Sets all plots to MVF view."""
        self.dataViewIndex = 0
        log.info("Data plots set to MVF view.")

        self.setDataView("view", self.dataViewIndex)
        # if self.dataLoaded:
        #     self.setRawDataView(self.dataViewIndex)

    def setIntensityView(self):
        """Sets all plots to intensity view."""
        self.dataViewIndex = 1
        log.info("Data plots set to intensity view.")
        if self.dataLoaded:
            self.setRawDataView(self.dataViewIndex)

    def changePlot2(self, selectedModels):
        """Updates plot 2 to show newly selected models to display.
        Args:
            selectedModels: List of string containing names of model/metric
                combinations that are selected in tab 2.
        """
        self.selectedModelNames = selectedModels
        self.updateUI()

    def updateUI(self):
        """Updates plots, tables, side menus.

        Should be called explicitly.
        """
        self.setDataView("view", self.dataViewIndex)

    #endregion

    #region Estimation, allocation
    def runModels(self, modelDetails):
        """Begins running estimation using selected models metrics.

        Args:
            modelDetails : A dict of models and metrics to use for
                calculations. List of model names as strings are one dict
                value, list of metric names as strings are other dict value.
        """
        # disable buttons until estimation complete
        self._main.tab1.sideMenu.runButton.setEnabled(False)
        self._main.tab4.sideMenu.allocationButton.setEnabled(False)
        modelsToRun = modelDetails["modelsToRun"]
        metricNames = modelDetails["metricNames"]
        if self.data:
            self.estimationComplete = False # estimation not complete since it just started running
            self._main.tab2.sideMenu.modelListWidget.clear()    # clear tab 2 list containing
                                                                # previously computed models,
                                                                # only added when calculations complete
            self._main.tab4.sideMenu.modelListWidget.clear()
            self.computeWidget = ComputeWidget(modelsToRun, metricNames, self.data)
            # DON'T WANT TO DISPLAY RESULTS IN ANOTHER WINDOW
            # WANT TO DISPLAY ON TAB 2/3
            self.computeWidget.results.connect(self.onEstimationComplete)   # signal emitted when estimation complete

    def onEstimationComplete(self, results):
        """
        description to be created at a later time

        Args:
            results: A dict containing model objects of model/metric
                combinations that estimation run on, indexed by name of
                combination as a string.
        """
        self.estimationComplete = True
        self.estimationResults = results
        self._main.tab1.sideMenu.runButton.setEnabled(True)  # re-enable button, can run another estimation
        self._main.tab4.sideMenu.allocationButton.setEnabled(True)  # re-enable allocation button, can't run
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

        log.info("DID NOT CONVERGE: %s", nonConvergedNames)

        self._main.tab2.sideMenu.addSelectedModels(convergedNames)  # add models to tab 2 list
                                                                    # so they can be selected
        # show which models didn't converge
        # self._main.tab2.sideMenu.addNonConvergedModels(nonConvergedNames)
        self._main.tab3.addResultsToTable(results)
        self._main.tab4.sideMenu.addSelectedModels(convergedNames)  # add models to tab 4 list so they
                                                                    # can be selected for allocation
        log.debug("Estimation results: %s", results)
        log.info("Estimation complete.")

    def runGoodnessOfFit(self):
        """Adds goodness of fit measures from estimation to tab 3 table."""
        if self.estimationComplete:
            self._main.tab3.addResultsToTable(self.estimationResults)

    def runAllocation(self, combinations):
        """Runs effort allocation on selected model/metric combinations.

        Args:
            combinations: List of model/metric combination names as strings.
        """
        B = self._main.tab4.sideMenu.budgetSpinBox.value()  # budget
        f = self._main.tab4.sideMenu.failureSpinBox.value()  # number of failures (UNUSED)

        self.allocationResults = {}  # create a dictionary for allocation results
        for i in range(len(combinations)):
            name = combinations[i]
            if " (No covariates)" not in name:
                m = self.estimationResults[name]  # model indexed by the name
                self.allocationResults[name] = [EffortAllocation(m, B, f), m]

        self._main.tab4.addResultsToTable(self.allocationResults, self.data)

    def runPrediction(self, failures):
        """Runs predictions for future points according to model results.

        Called when failure spin box value is changed.

        Args:
            failures: Number of future failure points to predict (int).
        """
        # run prediction on currently selected combinations in tab 2
        itemsSelected = len(self._main.tab2.sideMenu.modelListWidget.selectedItems())
        
        # check to make sure that model combinations are selected before
        # running prediction
        if self.estimationComplete and itemsSelected > 0:
            # gets first selected item
            name = self._main.tab2.sideMenu.modelListWidget.selectedItems()[0].text()
            m = self.estimationResults[name]  # model indexed by the name
            x, mvf_array, intensity_array = m.prediction(failures)

            # self.plotSettings.addLine(self.ax2, x, mvfList, "Prediction")
            # MVF view
            if self.dataViewIndex == 0:
                self.ax2.lines[-1].set_xdata(x)
                self.ax2.lines[-1].set_ydata(mvf_array)
            # Intensity view
            elif self.dataViewIndex == 1:
                self.ax2.lines[-1].set_xdata(x)
                self.ax2.lines[-1].set_ydata(intensity_array)

            # redraw figure
            self.ax2.legend()
            self.redrawPlot(2)

    #endregion


class MainWidget(QWidget):
    """Main UI widget of MainWindow class.

    Attributes:
        tabs: QTabWidget object containing the main tabs of the application.
        tab1: QWidget object containing UI elements for tab 1.
        tab2: QWidget object containing UI elements for tab 2.
        tab3: QWidget object containing UI elements for tab 3.
        tab4: QWidget object containing UI elements for tab 4.
    """

    def __init__(self):
        """Initializes main widget object."""
        super().__init__()
        self._initUI()

    def _initUI(self):
        """Initializes main widget UI elements."""
        layout = QVBoxLayout()
        # self.tabs = Tabs()

        self._initTabs()

        layout.addWidget(self.tabs)
        self.setLayout(layout)

    def _initTabs(self):
        """Creates main tabs and adds them to tab widget."""
        self.tabs = QTabWidget()

        self.tab1 = Tab1()
        self.tabs.addTab(self.tab1, "Data Upload and Model Selection")

        self.tab2 = Tab2()
        self.tabs.addTab(self.tab2, "Model Results and Predictions")

        self.tab3 = Tab3()
        self.tabs.addTab(self.tab3, "Model Comparison")

        self.tab4 = Tab4()
        self.tabs.addTab(self.tab4, "Effort Allocation")

        self.tabs.resize(300, 200)
