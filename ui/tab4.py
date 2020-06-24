# For handling debug output
import logging as log

# PyQt5 imports for UI elements
from PyQt5.QtWidgets import QWidget, QMessageBox, QHBoxLayout, QVBoxLayout, QLabel, \
                            QGroupBox, QListWidget, QPushButton, QAbstractItemView, \
                            QTableWidget, QTableWidgetItem, QAbstractScrollArea, \
                            QSpinBox, QDoubleSpinBox, QHeaderView, QRadioButton, \
                            QSpacerItem, QSizePolicy
from PyQt5.QtCore import pyqtSignal


class Tab4(QWidget):
    """Contains all widgets displayed on tab 4.

    Attributes:
        sideMenu: SideMenu object holding tab 3 widgets and their signals.
        table: QTableWidget that contains the goodness-of-fit measures for each
            calculated model/metric combination.
        font: QFont object that is formatted bold. Used to set text bold for
            cells containing the highest ranked combinations, according to the
            weighting of each measure.
    """

    def __init__(self):
        """Initializes tab 4 UI elements."""
        super().__init__()
        self._setupTab4()

    def addResultsToTable(self, results, data):
        """Adds effort allocation results to the tab 4 table.

        Args:
            results: A dict containing a list of the effort allocation results
                and model objects as values. Indexed by the name of the
                model/metric combination.
                results[name] = [EffortAllocation, Model]
            data: Data object contiaining imported data as a Pandas dataframe.
        """
        self.table.setSortingEnabled(False)  # disable sorting while editing contents
        self.table.clear()
        self.table.setColumnCount(3 + len(data.metricNames))
        self.table.setHorizontalHeaderLabels(self._createHeaderLabels(data.metricNames))
        self.table.setRowCount(len(results))    # set row count to include all model results, 
                                                # even if not converged
        row = 0   # rows

        for key, value in results.items():
            res = value[0]
            model = value[1]

            self.table.setItem(row, 0, QTableWidgetItem(model.shortName))   # model name
            self.table.setItem(row, 1, QTableWidgetItem(model.metricString))  # model metrics
            self.table.setItem(row, 2, QTableWidgetItem("{0:.2f}".format(res.H)))
            # number of columns = number of covariates
            j = 0
            for name in model.metricNames:
                col = data.metricNameDictionary[name]
                self.table.setItem(row, 3 + col, QTableWidgetItem("{0:.2f}".format(res.percentages[j])))
                j += 1
            row += 1

        self.table.setRowCount(row)   # set row count to only include converged models
        self.table.resizeColumnsToContents()    # resize column width after table is edited
        self.table.setSortingEnabled(True)      # re-enable sorting after table is edited

    def _setupTab4(self):
        """Creates tab 4 widgets and adds them to layout."""
        mainLayout = QHBoxLayout()  # main tab layout

        self.sideMenu = SideMenu4()
        mainLayout.addLayout(self.sideMenu, 15)
        self.table = self._setupTable()
        mainLayout.addWidget(self.table, 85)
        self.setLayout(mainLayout)

    def _setupTable(self):
        """Creates table widget with proper headers.

        Returns:
            A QTableWidget with specified column headers.
        """
        table = QTableWidget()
        table.setEditTriggers(QTableWidget.NoEditTriggers)  # make cells unable to be edited
        table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
                                                            # column width fit to contents
        table.setRowCount(1)
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Model Name", "Covariates", "Estimated failures"])

        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        # provides bottom border for header
        stylesheet = "::section{Background-color:rgb(250,250,250);}"
        header.setStyleSheet(stylesheet)
        table.move(0, 0)

        return table

    def _createHeaderLabels(self, metricNames):
        """Creates the header labels for the tab 4 table.

        The header labels are based on the names of the covariate metrics
        from the data. The effort allocation shows the percent of the budget
        that should be allocated for each metric, so a percent sign (%) is
        added before each metric name.

        Args:
            metricNames: A list of the names of the covariate metrics as
                strings.

        Returns:
            A list of column headers as strings, including the specifed metric
            names.
        """
        percentNames = []
        i = 0
        for name in metricNames:
            percentNames.append("%" + name)
            i += 1
        headerLabels = ["Model Name", "Covariates", "H"] + percentNames
        return headerLabels


class SideMenu4(QVBoxLayout):
    """Side menu for tab 4.

    Attributes:
        allocationButton: QPushButton object, begins effort allocation when
            clicked.
        modelListWidget: QListWidget containing the names of converged
            model/metric combinations.
        budgetSpinBox: QDoubleSpinBox widget, specifies the budget used when
            performing the effort allocation calculation.
        failureSpinBox: QSpinBox widget, specifies the desired number of
            failures to detect for effort allocation calculation.
        runAllocationSignal: pyqtSignal, emits list of combination names to run
            the effort allocation on.
    """

    # signals
    runAllocationSignal = pyqtSignal(list)  # starts allocation computation

    def __init__(self):
        """Initializes tab 4 side menu UI elements."""
        super().__init__()
        self._setupSideMenu()

    def addSelectedModels(self, modelNames):
        """Creates list of model/metric combination names that include covariates.

        Results with no covariates are not added to the list, since allocation
        can only be performed on combinations with covariate metrics. Called
        when estimation is complete.

        Args:
            modelNames: A list of strings, contains the name of each
                model/metric combination that converged.
        """

        for name in modelNames:
            if " (None)" not in name:
                self.modelListWidget.addItem(name)

    def _setupSideMenu(self):
        """Creates group box widgets and adds them to layout."""
        modelsGroup = QGroupBox("Select Models/Metrics for Allocation")
        modelsGroup.setLayout(self._setupModelsGroup())
        optionsGroup = QGroupBox("Allocation Parameters")
        optionsGroup.setLayout(self._setupOptionsGroup())
        self._setupAllocationButton()

        self.addWidget(modelsGroup, 10)
        self.addWidget(optionsGroup)
        self.addWidget(self.allocationButton, 1)

        self.addStretch(1)

    def _setupModelsGroup(self):
        """Creates widget containing list of converged model/metric combos.

        Returns:
            A created VBoxLayout containing the created models group.
        """
        modelGroupLayout = QVBoxLayout()
        self.modelListWidget = QListWidget()
        modelGroupLayout.addWidget(self.modelListWidget)
        self.modelListWidget.setSelectionMode(QAbstractItemView.MultiSelection)  # able to select multiple models

        return modelGroupLayout

    def _setupOptionsGroup_old(self):
        """Creates widgets for specifying effort allocation parameters.

        Returns:
            A created VBoxLayout containing the created options group.
        """
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

    def _setupOptionsGroup(self):
        """Creates widgets for specifying effort allocation parameters.

        Returns:
            A created VBoxLayout containing the created options group.
        """

        optionsGroupLayout = QVBoxLayout()
        tempBudget = QHBoxLayout()
        tempFailures = QHBoxLayout()
        budgetVertical = QVBoxLayout()
        failuresVertical = QVBoxLayout()

        budgetVertical.addWidget(QLabel("Budget"))
        self.budgetSpinBox = QDoubleSpinBox()
        # self.budgetSpinBox.setMaximumWidth(200)
        self.budgetSpinBox.setRange(0.0, 999999.0)
        self.budgetSpinBox.setValue(20)
        budgetVertical.addWidget(self.budgetSpinBox)

        tempBudget.addWidget(QRadioButton())
        tempBudget.addLayout(budgetVertical, 1)

        failuresVertical.addWidget(QLabel("Failures"))
        self.failureSpinBox = QSpinBox()
        # self.failureSpinBox.setMaximumWidth(200)
        self.failureSpinBox.setRange(1, 999999)
        failuresVertical.addWidget(self.failureSpinBox)

        tempFailures.addWidget(QRadioButton())
        tempFailures.addLayout(failuresVertical,1)

        optionsGroupLayout.addLayout(tempBudget)
        vspacer = QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)
        optionsGroupLayout.addItem(vspacer)
        optionsGroupLayout.addLayout(tempFailures)

        return optionsGroupLayout

    def _setupAllocationButton(self):
        """Creates the button that begins effort allocation."""
        self.allocationButton = QPushButton("Run Allocation")
        self.allocationButton.setEnabled(False) # begins disabled since no model has been run yet
        # self.allocationButton.setMaximumWidth(250)
        self.allocationButton.clicked.connect(self._emitRunAllocationSignal)

    def _emitRunAllocationSignal(self):
        """Emits signal that effort allocation with model/metric combiations.

        Method called when Run Allocation button is pressed. The emitted signal
        (runAllocationSignal) contains the list of combinations to run the
        allocation on. The signal is only emitted if at least one combination
        is selected.
        """
        selectedCombinationNames = [item.text() for item in self.modelListWidget.selectedItems()]
        if selectedCombinationNames:
            selectedCombinationNames = [item.text() for item in self.modelListWidget.selectedItems()]
            log.info("Selected for Allocation: %s", selectedCombinationNames)
            self.runAllocationSignal.emit(selectedCombinationNames)

        # if no models/metric combinations selected, create message box to
        # display this warning

        else:
            log.warning("Must select at least one model/metric combination for allocation.")
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setText("No selection made for allocation")
            msgBox.setInformativeText("Please select at least one model/metric combination.")
            msgBox.setWindowTitle("Warning")
            msgBox.exec_()
