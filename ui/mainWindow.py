"""
Contains highest level UI elements. Connects all core modules and functions to
the UI elements. Able to reference all elements and the signals they emit.
"""


# For handling debug output
import logging as log

# for handling fonts
import sys

# PyQt5 imports for UI elements
from PyQt5.QtWidgets import QMainWindow, qApp, QWidget, QTabWidget, \
                            QVBoxLayout, QAction, QActionGroup, QFileDialog
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QIcon

# Local imports
import models
from ui.commonWidgets import ComputeWidget, PSSEThread
from ui.tab1 import Tab1
from ui.tab2 import Tab2
from ui.tab3 import Tab3
from ui.tab4 import Tab4
from core.dataClass import Data
from core.allocation import EffortAllocation
from core.goodnessOfFit import PSSE
import core.prediction as prediction


class MainWindow(QMainWindow):
    """Window that is displayed when starting application.

    Provides top level control of application. Connects model functions and
    UI elements through signal connections. Handles file opening, running
    estimation/allocation/trend tests, creating/updating plots, menu options.

    Attributes:
        _main: Instance of MainWidget class, contains widgets.
        debug: Boolean indicating if debug mode is active or not.
        data: Pandas dataframe containing imported data.
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
        """
        Initializes MainWindow, not in debug mode by default.

        Args
            debug: Boolean indicating if debug mode is activated
        """
        super().__init__()

        self._main = MainWidget()
        self.setCentralWidget(self._main)

        # set debug mode
        self.debug = debug

        # set data
        self.data = Data()
        self.selectedModelNames = []

        # flags
        self.dataLoaded = False
        self.estimationComplete = False

        # SIGNAL CONNECTIONS
        self.importFileSignal.connect(self.importFile)
        self._main.tab1.sideMenu.sheetChangedSignal.connect(self.changeSheet)
        self._main.tab1.sideMenu.sliderSignal.connect(self.subsetData)
        # run models when signal is received
        self._main.tab1.sideMenu.runModelSignal.connect(self.runModels)
        self._main.tab2.sideMenu.modelChangedSignal.connect(self.changePlot2AndUpdateComparisonTable)
        # connect tab2 list changed to refreshing tab 2 plot
        self._main.tab2.sideMenu.failureChangedSignal.connect(self.updatePredictionPlotMVF)
        self._main.tab2.sideMenu.intensityChangedSignal.connect(self.updatePredictionPlotIntensity)
        self._main.tab3.sideMenu.modelChangedSignal.connect(self.changePlot2AndUpdateComparisonTable)
        self._main.tab3.sideMenu.runPSSESignal.connect(self.runPSSE)
        self._main.tab3.sideMenu.spinBoxChangedSignal.connect(self.runGoodnessOfFit)
        self._main.tab4.sideMenu.runAllocation1Signal.connect(self.runAllocation1)
        self._main.tab4.sideMenu.runAllocation2Signal.connect(self.runAllocation2)

        self._initUI()
        log.info("UI loaded.")

    def _initUI(self):
        """Sets window parameters, fonts, initializes UI elements."""
        # setup main window parameters
        title = "C-SFRAT"
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
        self.setWindowIcon(QIcon('ui/C-SFRAT_logo_256.png'))

        # 0 indicates MVF plot, 1 indicates intensity plot
        self.plotViewIndex = 0

        # setup font for entire application
        if sys.platform == "win32":
        	# windows
        	self.setStyleSheet('QWidget {font: 12pt "Segoe"}')
        elif sys.platform == "darwin":
        	# macos
        	self.setStyleSheet('QWidget {font: 12pt "Verdana"}')
       	elif sys.platform == "linux" or sys.platform == "linux2":
       		# linux
       		self.setStyleSheet('QWidget {font: 12pt "Arial"}')

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
        openFile.setStatusTip("Import data file")
        openFile.triggered.connect(self.fileOpened)

        # export table (tab 2)
        exportTable2 = QAction("Export Table (Tab 2)", self)
        # exportTable.setShortcut("Ctrl+E")
        exportTable2.setStatusTip("Export tab 2 table to csv")
        exportTable2.triggered.connect(self.exportTable2)

        # export table (tab 3)
        exportTable3 = QAction("Export Table (Tab 3)", self)
        # exportTable3.setShortcut("Ctrl+E")
        exportTable3.setStatusTip("Export tab 3 table to csv")
        exportTable3.triggered.connect(self.exportTable3)

        # exit
        exitApp = QAction("Exit", self)
        exitApp.setShortcut("Ctrl+Q")
        exitApp.setStatusTip("Close application")
        exitApp.triggered.connect(self.closeEvent)

        # add actions to file menu
        fileMenu.addAction(openFile)
        fileMenu.addSeparator()
        fileMenu.addAction(exportTable2)
        fileMenu.addAction(exportTable3)
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

        # smooth
        smooth = QAction("Smooth Plot (Fitted Models)", self, checkable=True)
        smooth.setShortcut("Ctrl+F")
        smooth.setStatusTip("Fitted model plot shows smooth curves")
        smooth.setChecked(True)
        smooth.triggered.connect(self.setSmoothPlot)
        lineStyle.addAction(smooth)

        # step
        step = QAction("Step Plot (Fitted Models)", self, checkable=True)
        step.setShortcut("Ctrl+D")
        step.setStatusTip("Fitted model plot shown as step")
        step.triggered.connect(self.setStepPlot)
        lineStyle.addAction(step)

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

        # add actions to view menu
        viewMenu.addSeparator()
        viewMenu.addActions(graphStyle.actions())

    def closeEvent(self, event):
        """
        Quits all threads, and shuts down app.

        Called when application is closed by user. Waits to abort symbolic and
        estimation threads safely if they are still running when application
        is closed.
        """
        log.info("Covariate Tool application closed.")

        # --- stop running threads ---
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
        True afterwards.
        """
        # default location is datasets directory
        files = QFileDialog.getOpenFileName(self, "Open profile", "datasets",
                                            filter=("Data Files (*.csv *.xls *.xlsx)"))
        # if a file was loaded
        if files[0]:
            self.data.importFile(files[0])  # imports loaded file
            self.dataLoaded = True
            log.info("Data loaded from %s", files[0])
            self.importFileSignal.emit()    # emits signal that file was
                                            # imported successfully

    def importFile(self):
        """Sets UI elements with imported data.

        Updates sheet select on tab 1 with sheet names (if applicable). Calls
        setDataView method to update tab 1 plot and table.
        """
        # clear sheet names from previous file
        self._main.tab1.sideMenu.sheetSelect.clear()
        # add sheet names from new file
        self._main.tab1.sideMenu.addSheets(self.data.sheetNames)
        # add spin boxes to tab 2 for each covariate, used for prediction
        self._main.tab2.sideMenu.updateEffortList(self.data.metricNames)
        self.changeSheet(0)     # always show first sheet when loaded

    def changeSheet(self, index):
        """Changes the current sheet displayed.
        Handles data that needs to be changed when sheet changes.

        Args:
            index: The index of the sheet (int).
        """
        self.data.currentSheet = index      # store
        self.data.max_interval = self.data.n
        self._main.tab1.sideMenu.updateSlider(self.data.n)

        self.createPlots()

        self._main.tab1.updateTable(self.data.getData())

        # display either MVF or intensity plots, depending on what is currently selected
        self._main.tab1.plotAndTable.plotWidget.changePlotType(self.plotViewIndex)
        self._main.tab2.plotAndTable.plotWidget.changePlotType(self.plotViewIndex)

        self.setMetricList()

    def createPlots(self):
        """
        Called when data is loaded. Creates step/bar plots displaying imported data.
        """

        # tab 1 plots
        x = self.data.getData()['T']    # time vector
        cfc = self.data.getData()['CFC']    # cumulative failure count vector
        fc = self.data.getData()['FC']  # failure count vector

        self._main.tab1.plotAndTable.plotWidget.createPlots(x, cfc, fc)

        # tab 2 plots
        self._main.tab2.plotAndTable.plotWidget.createPlots(x, cfc, fc)

    def setMetricList(self):
        """Updates tab 1 list widget with metric names on current sheet."""
        self._main.tab1.sideMenu.metricListWidget.clear()
        if self.dataLoaded:
            # data class stores all combinations of metric names
            self._main.tab1.sideMenu.metricListWidget.addItems(self.data.metricNameCombinations)
            log.info("%d covariate metrics on this sheet: %s", self.data.numCovariates,
                                                               self.data.metricNames)

    def subsetData(self, slider_value):
        # minimum subset is 5 data points
        if slider_value < 5:
            self._main.tab1.sideMenu.slider.setValue(5)
        self.data.max_interval = slider_value

        ## new max interval, getData returns subset
        ## mvf plots
        x = self.data.getData()['T']   # setData does not accept dataframes or np arrays for some reason
        y_mvf = self.data.getData()['CFC']
        y_intensity = self.data.getData()['FC']

        self._main.tab1.plotAndTable.plotWidget.subsetPlots(x, y_mvf, y_intensity)
        self._main.tab2.plotAndTable.plotWidget.subsetPlots(x, y_mvf, y_intensity)

    def redrawPlot(self, tabNumber):
        """Redraws plot for the provided tab number.

        Args:
            tabNumber: Tab number (int) that contains the figure to redraw.
        """
        if tabNumber == 1:
            self._main.tab1.plotAndTable.figure.canvas.draw()
        elif tabNumber == 2:
            self._main.tab2.plotAndTable.figure.canvas.draw()

    #region plot styles
    def setLineView(self):
        """Sets plot style to line."""
        # self.setPlotStyle(style='-')
        self._main.tab1.plotAndTable.plotWidget.setLineView()
        self._main.tab2.plotAndTable.plotWidget.setLineView()
        log.info("Plot style set to line view.")

    def setPointsView(self):
        """Sets plot style to points."""
        # self.setPlotStyle(style='o')
        self._main.tab1.plotAndTable.plotWidget.setPointsView()
        self._main.tab2.plotAndTable.plotWidget.setPointsView()
        log.info("Plot style set to points view.")

    def setLineAndPointsView(self):
        """Sets plot style to line and points."""
        # self.setPlotStyle(style='-o')
        self._main.tab1.plotAndTable.plotWidget.setLineAndPointsView()
        self._main.tab2.plotAndTable.plotWidget.setLineAndPointsView()
        log.info("Plot style set to line and points view.")
    #endregion

    #region plot types
    def setStepPlot(self):
        """Sets plot type to step plot."""
        self._main.tab1.plotAndTable.plotWidget.setStepPlot()
        self._main.tab2.plotAndTable.plotWidget.setStepPlot()
        log.info("Line style set to 'step'.")

    def setSmoothPlot(self):
        """Sets plot type to smooth line ('plot')"""
        self._main.tab1.plotAndTable.plotWidget.setSmoothPlot()
        self._main.tab2.plotAndTable.plotWidget.setSmoothPlot()
        log.info("Line style set to 'smooth'.")
    #endregion

    def setMVFView(self):
        """Sets all plots to MVF view."""
        self.plotViewIndex = 0
        log.info("Data plots set to MVF view.")

        if self.dataLoaded:
            self._main.tab1.plotAndTable.plotWidget.changePlotType(self.plotViewIndex)
            self._main.tab2.plotAndTable.plotWidget.changePlotType(self.plotViewIndex)

        if self.estimationComplete:
            # disable intensity spin box
            self._main.tab2.sideMenu.reliabilitySpinBox.setDisabled(True)
            # enable failure spin box
            self._main.tab2.sideMenu.failureSpinBox.setEnabled(True)

            # tab 2 table shows MVF
            self._main.tab2.setTableModel(0)

    def setIntensityView(self):
        """Sets all plots to intensity view."""
        self.plotViewIndex = 1
        log.info("Data plots set to intensity view.")
        # if self.dataLoaded:
        #     self.setRawDataView(self.plotViewIndex)

        if self.dataLoaded:
            self._main.tab1.plotAndTable.plotWidget.changePlotType(self.plotViewIndex)
            self._main.tab2.plotAndTable.plotWidget.changePlotType(self.plotViewIndex)

        if self.estimationComplete:
            # disable failure spin box
            self._main.tab2.sideMenu.failureSpinBox.setDisabled(True)
            # enable intensity spin box
            self._main.tab2.sideMenu.reliabilitySpinBox.setEnabled(True)

            # tab 2 table shows intensity
            self._main.tab2.setTableModel(1)

    def changePlot2AndUpdateComparisonTable(self, selectedModels):
        # Access Selected Items
        # Find which tab the change came from
        # Disable the signals
        # Change the other tab
        # Change both plot and table
        # self._main.tab2.sideMenu.modelList.selectedItems()
        ModelsList2 = self._main.tab2.sideMenu.modelListWidget
        ModelsList3 = self._main.tab3.sideMenu.modelListWidget
        Modelstext = self._main.tab2.sideMenu.ModelsText

        ModelsList2.blockSignals(True)
        ModelsList3.blockSignals(True)

        for i in Modelstext:
            if i in selectedModels:
                ModelsList3.item(Modelstext.index(i)).setSelected(True)
                ModelsList2.item(Modelstext.index(i)).setSelected(True)
            else:
                ModelsList3.item(Modelstext.index(i)).setSelected(False)
                ModelsList2.item(Modelstext.index(i)).setSelected(False)

        ModelsList2.blockSignals(False)
        ModelsList3.blockSignals(False)

        selectedNums = [x.split('. ', 1)[0] for x in selectedModels]
        selectedNames = [x.split('. ', 1)[1] for x in selectedModels]

        self.updateComparisonTable(selectedNums, selectedNames)

        self._main.tab2.plotAndTable.plotWidget.updateLines(selectedNames)

        # pass selected nums to tab 2 tableView
        # this way, we can show/hide columns depending on what is selected
        self._main.tab2.updateTableView(selectedNums)

        self.selectedModelNames = selectedNames

    def updateComparisonTable(self, selectedNums, selectedNames):

        self._main.tab3.updateTableView(selectedNums)
        self.selectedModelNums = selectedNums

    def changePlot2(self, selectedModels):
        """Updates plot 2 to show newly selected models to display.
        Args:
            selectedModels: List of string containing names of model/metric
                combinations that are selected in tab 2.
        """
        self.selectedModelNames = selectedModels
        self.updateUI()

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
        self._main.tab1.sideMenu.runButton.setDisabled(True)
        self._main.tab3.sideMenu.psseSpinBox.setDisabled(True)
        self._main.tab4.sideMenu.allocation1Button.setDisabled(True)
        self._main.tab4.sideMenu.allocation2Button.setDisabled(True)
        modelsToRun = modelDetails["modelsToRun"]
        metricNames = modelDetails["metricNames"]


        # ******* NEED TO CLEAR PLOTS AND TABLES *******

        if self.data:
            self.estimationComplete = False # estimation not complete since it just started running
            self.psseComplete = False       # must re-run PSSE after fitting new models
            self.selectedModelNames = []    # clear selected models, since none are selected when new models are fitted

            # need to block signals so update signal doesn't fire when list widgets are cleared
            self._main.tab2.sideMenu.modelListWidget.blockSignals(True)
            self._main.tab3.sideMenu.modelListWidget.blockSignals(True)
            self._main.tab2.sideMenu.modelListWidget.clear()    # clear tab 2 list containing
                                                                # previously computed models,
                                                                # only added when calculations complete
            self._main.tab3.sideMenu.modelListWidget.clear()
            self._main.tab4.sideMenu.modelListWidget.clear()

            self.computeWidget = ComputeWidget(modelsToRun, metricNames, self.data)
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

        # run PSSE on data along with model fitting
        self.runPSSE(self._main.tab3.sideMenu.psseParameterSpinBox.value())

        self._main.tab1.sideMenu.runButton.setEnabled(True)  # re-enable button, can run another estimation
        # self._main.tab3.sideMenu.psseButton.setEnabled(True)    # enable PSSE button now that we have fitted models
        self._main.tab4.sideMenu.allocation1Button.setEnabled(True)     # re-enable allocation buttons, can't run
        self._main.tab4.sideMenu.allocation2Button.setEnabled(True)     # if estimation not complete

        if self.plotViewIndex == 0:
            # enable failure spin box
            self._main.tab2.sideMenu.failureSpinBox.setEnabled(True)
        elif self.plotViewIndex == 1:
            # enable intensity spin box
            self._main.tab2.sideMenu.reliabilitySpinBox.setEnabled(True)

        # create lines for each plot
        self._main.tab1.plotAndTable.plotWidget.createLines(results)
        self._main.tab2.plotAndTable.plotWidget.createLines(results)

        convergedNames = []
        nonConvergedNames = []
        for key, model in results.items():
            if model.converged:
                convergedNames.append(key)
            else:
                nonConvergedNames.append(key)

        log.info("DID NOT CONVERGE: %s", nonConvergedNames)

        for i in range(1, len(convergedNames)+1):
            convergedNames[i-1] = "{0}. {1}".format(i, convergedNames[i-1])

        self._main.tab2.sideMenu.addSelectedModels(convergedNames)  # add models to tab 2 list
                                                                    # so they can be selected

        self._main.tab2.updateModel(self.estimationResults)

        self._main.tab3.updateModel(self.estimationResults)   # add converged results to model containing all result data
        self._main.tab3.sideMenu.addSelectedModels(convergedNames)  # add models to tab 3 list
                                                                    # so they can be selected for comparison
        # self._main.tab3.addResultsToTable(results)
        self._main.tab4.sideMenu.addSelectedModels(convergedNames)  # add models to tab 4 list so they
                                                                    # can be selected for allocation
        # can re-enable signals for list widgets now
        self._main.tab2.sideMenu.modelListWidget.blockSignals(False)
        self._main.tab3.sideMenu.modelListWidget.blockSignals(False)
        log.debug("Estimation results: %s", results)
        log.info("Estimation complete.")

    def runGoodnessOfFit(self):
        """Adds goodness of fit measures from estimation to tab 3 table."""

        if self.estimationComplete:
            combinations = [item.text() for item in self._main.tab3.sideMenu.modelListWidget.selectedItems()]

            selectedNums = [x.split('. ', 1)[0] for x in combinations]
            selectedNames = [x.split('. ', 1)[1] for x in combinations]

            self.updateComparisonTable(selectedNums, selectedNames)

    def runAllocation1(self, combinations):
        """Runs effort allocation on selected model/metric combinations.

        Args:
            combinations: List of model/metric combination names as strings.
        """
        B = self._main.tab4.sideMenu.budgetSpinBox.value()  # budget

        self.allocationResults = {}  # create a dictionary for allocation results
        for i in range(len(combinations)):
            name = combinations[i]
            if " (No covariates)" not in name:
                m = self.estimationResults[name]  # model indexed by the name

                ## RUN PREDICTION USING SPECIFIED SUBSET OF COVARIATE DATA
                self.allocationResults[name] = [EffortAllocation(m, m.covariateData, 1, B), m]

        self._main.tab4.addResultsToTable(self.allocationResults, self.data, 1)

    def runAllocation2(self, combinations):
        """Runs effort allocation on selected model/metric combinations.

        Args:
            combinations: List of model/metric combination names as strings.
        """
        f = self._main.tab4.sideMenu.failureSpinBox.value()  # number of failures

        self.allocationResults = {}  # create a dictionary for allocation results
        for i in range(len(combinations)):
            name = combinations[i]
            if " (No covariates)" not in name:
                m = self.estimationResults[name]  # model indexed by the name

                ## RUN PREDICTION USING SPECIFIED SUBSET OF COVARIATE DATA
                self.allocationResults[name] = [EffortAllocation(m, m.covariateData, 2, f), m]

        # just add to table 2
        self._main.tab4.addResultsToTable(self.allocationResults, self.data, 2)

    def updatePredictionPlotMVF(self):
        if self.estimationComplete:
            # TEMPORARY
            # for displaying predictions in tab 2 table
            prediction_list = [0]
            model_name_list = ["Interval"]

            # check if prediction is specified
            if self._main.tab2.sideMenu.failureSpinBox.value() > 0:

                for key, model in self.estimationResults.items():
                    # add line for model if selected
                    # model = self.estimationResults[modelName]
                
                    x, mvf_array = self.runPredictionMVF(model, self._main.tab2.sideMenu.failureSpinBox.value())
                    self._main.tab2.plotAndTable.plotWidget.updateLineMVF(key, x, mvf_array)

                    ## TABLE
                    prediction_list[0] = x
                    prediction_list.append(mvf_array)
                    model_name_list.append(model.combinationName)
                    
            else:
                # set plot and table back to model with no predictions
                for key, model in self.estimationResults.items():
                    self._main.tab2.plotAndTable.plotWidget.updateLineMVF(key, model.t, model.mvf_array)

                    prediction_list[0] = model.t
                    prediction_list.append(model.mvf_array)
                    model_name_list.append(model.combinationName)

            # 0 indicates MVF
            self._main.tab2.updateTable_prediction(prediction_list, model_name_list, 0)


    def updatePredictionPlotIntensity(self):
        if self.estimationComplete:
            # for displaying predictions in tab 2 table
            prediction_list = [0]
            model_name_list = ["Interval"]
            max_x = 0

            # check if prediction is specified
            if self._main.tab2.sideMenu.reliabilitySpinBox.value() > 0.0:

                for key, model in self.estimationResults.items():
                    x, intensity_array, interval = self.runPredictionIntensity(model, self._main.tab2.sideMenu.reliabilitySpinBox.value())
                    self._main.tab2.plotAndTable.plotWidget.updateLineIntensity(key, x, intensity_array)


                    ## TABLE
                    # make sure we don't get NaN for interval column
                    # otherwise, if the last combination does not have the most intervals,
                    # then the remaining values of the interval column will be NaN since
                    # they were not defined
                    if len(x) > max_x:
                        prediction_list[0] = x
                        max_x = len(x)

                    prediction_list.append(intensity_array)
                    model_name_list.append(model.combinationName)
                    
            else:
                for key, model in self.estimationResults.items():
                    self._main.tab2.plotAndTable.plotWidget.updateLineIntensity(key, model.t, model.intensityList)

                    prediction_list[0] = model.t
                    prediction_list.append(model.intensityList)
                    model_name_list.append(model.combinationName)


            # 1 indicates intensity
            self._main.tab2.updateTable_prediction(prediction_list, model_name_list, 1)


    def runPredictionMVF(self, model, failures):
        """Runs predictions for future points according to model results.

        Called when failure spin box value is changed.

        Args:
            failures: Number of future failure points to predict (int).
        """

        # m = self.estimationResults[modelName]  # model indexed by the name

        x, mvf_array = prediction.prediction_mvf(model, failures, model.covariateData, self._main.tab2.sideMenu.effortSpinBoxDict)

        return x, mvf_array#, intensity_array

    def runPredictionIntensity(self, model, intensity):
        x, intensity_array, intervals = prediction.prediction_intensity(model, intensity, model.covariateData, self._main.tab2.sideMenu.effortSpinBoxDict)
        return x, intensity_array, intervals

    def runPSSE(self, fraction):
        # determine subset of data (from UI elements)
        # minimum of 5 data points, max of n-1

        # perform model fitting on that subset
        # adapt TaskThread run() method?
        # pass specified subset of covariate data

        # goodnessOfFit.PSSE()

        """Begins running model fitting for PSSE.

        Args:
            modelDetails : A dict of models and metrics to use for
                calculations. List of model names as strings are one dict
                value, list of metric names as strings are other dict value.
        """
        
        if self.data:
            # disable PSSE button until model fitting completes
            self._main.tab3.sideMenu.psseButton.setDisabled(True)

            self.psseComplete = False

            modelsToRun = []
            metricNames = []

            for key, model in self.estimationResults.items():
                # need to get model classes, to instantiate new objects
                if models.modelList[model.__class__.__name__] not in modelsToRun:
                    modelsToRun.append(models.modelList[model.__class__.__name__])
                # only want the first instances of metrics, otherwise we'll get duplicates
                if model.metricNames not in metricNames:
                    metricNames.append(model.metricNames)

            self.psse_thread = PSSEThread(modelsToRun, metricNames, self.data, fraction)
            self.psse_thread.results.connect(self.onPSSEComplete)   # signal emitted when estimation complete
            self.psse_thread.start()

    def onPSSEComplete(self, results):
        """
        Called when PSSE thread is done running

        Args:
            results: A dict containing model objects of model/metric
                combinations that estimation run on, indexed by name of
                combination as a string.
        """
        
        self.psseResults = results
        self._main.tab3.addResultsPSSE(self.psseResults)
        self.psseComplete = True

        # re-enable PSSE button
        self._main.tab3.sideMenu.psseButton.setEnabled(True)
        # enable PSSE weight spinbox
        self._main.tab3.sideMenu.psseSpinBox.setEnabled(True)

    def exportTable2(self):
        path = QFileDialog.getSaveFileName(self,
            'Export model results', 'model_results.csv', filter='CSV (*.csv)')

        if path[0]:
            self._main.tab2.exportTable(path[0])

    def exportTable3(self):
        path = QFileDialog.getSaveFileName(self,
            'Export model results', 'model_results.csv', filter='CSV (*.csv)')

        if path[0]:
            self._main.tab3.exportTable(path[0])

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
