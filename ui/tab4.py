# For handling debug output
import logging as log

# PyQt5 imports for UI elements
from PyQt5.QtWidgets import QWidget, QMessageBox, QHBoxLayout, QVBoxLayout, QLabel, \
                            QGroupBox, QListWidget, QPushButton, QAbstractItemView, \
                            QTableWidget, QTableWidgetItem, QAbstractScrollArea, \
                            QSpinBox, QDoubleSpinBox, QHeaderView, QRadioButton, \
                            QSpacerItem, QSizePolicy, QTabWidget
from PyQt5.QtCore import pyqtSignal

#Temp Imports
##########################
from ui.commonWidgets import TableTabs
##########################

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

    def addResultsToTable(self, results, data, allocation_type):
        """Adds effort allocation results to the tab 4 table.

        Args:
            results: A dict containing a list of the effort allocation results
                and model objects as values. Indexed by the name of the
                model/metric combination.
                results[name] = [EffortAllocation, Model]
            data: Data object contiaining imported data as a Pandas dataframe.
        """

        if allocation_type == 1:
            # FOR BUDGET
            self.TableAndTable.budgetTab.setSortingEnabled(False)  # disable sorting while editing contents
            self.TableAndTable.budgetTab.clear()
            self.TableAndTable.budgetTab.setColumnCount(3 + len(data.metricNames))
            self.TableAndTable.budgetTab.setHorizontalHeaderLabels(self._createHeaderLabels(data.metricNames)[0])
            self.TableAndTable.budgetTab.setRowCount(len(results))      # set row count to include all model results,
                                                                        # even if not converged

            row = 0
            for key, value in results.items():
                res = value[0]
                model = value[1]

                self.TableAndTable.budgetTab.setItem(row, 0, QTableWidgetItem(model.shortName))   # model name
                self.TableAndTable.budgetTab.setItem(row, 1, QTableWidgetItem(model.metricString))  # model metrics
                self.TableAndTable.budgetTab.setItem(row, 2, QTableWidgetItem("{0:.2f}".format(res.H)))

                # number of columns = number of covariates
                j = 0
                for name in model.metricNames:
                    col = data.metricNameDictionary[name]
                    self.TableAndTable.budgetTab.setItem(row, 3 + col, QTableWidgetItem("{0:.2f}".format(res.percentages[j])))
                    j += 1
                row += 1

            self.TableAndTable.budgetTab.setRowCount(row)   # set row count to only include converged models
            self.TableAndTable.budgetTab.resizeColumnsToContents()    # resize column width after table is edited
            self.TableAndTable.budgetTab.setSortingEnabled(True)      # re-enable sorting after table is edited

        else:
            self.TableAndTable.failureTab.setSortingEnabled(False)  # disable sorting while editing contents
            self.TableAndTable.failureTab.clear()
            self.TableAndTable.failureTab.setColumnCount(3 + len(data.metricNames))
            self.TableAndTable.failureTab.setHorizontalHeaderLabels(self._createHeaderLabels(data.metricNames)[1])
            self.TableAndTable.failureTab.setRowCount(len(results))    # set row count to include all model results,
                                                    # even if not converged

            row = 0
            for key, value in results.items():
                res = value[0]
                model = value[1]
                self.TableAndTable.failureTab.setItem(row, 0, QTableWidgetItem(model.shortName))   # model name
                self.TableAndTable.failureTab.setItem(row, 1, QTableWidgetItem(model.metricString))  # model metrics
                self.TableAndTable.failureTab.setItem(row, 2, QTableWidgetItem("{0:.2f}".format(res.effort)))
                
                # number of columns = number of covariates
                j = 0
                for name in model.metricNames:
                    col = data.metricNameDictionary[name]
                    self.TableAndTable.failureTab.setItem(row, 3 + col, QTableWidgetItem("{0:.2f}".format(res.percentages2[j])))
                    #For failures tab, do : res.percetages2[]
                    j += 1
                row += 1

            self.TableAndTable.failureTab.setRowCount(row)   # set row count to only include converged models
            self.TableAndTable.failureTab.resizeColumnsToContents()    # resize column width after table is edited
            self.TableAndTable.failureTab.setSortingEnabled(True)      # re-enable sorting after table is edited

    def _setupTab4(self):
        """Creates tab 4 widgets and adds them to layout."""
        mainLayout = QHBoxLayout()  # main tab layout

        self.sideMenu = SideMenu4()
        mainLayout.addLayout(self.sideMenu, 15)

        self.TableAndTable = TableTabs("Allocation 1", "Allocation 2")
        mainLayout.addWidget(self.TableAndTable, 85)
        self.setLayout(mainLayout)

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
        headerLabels=[]
        percentNames = []
        i = 0
        for name in metricNames:
            percentNames.append("%" + name)
            i += 1
        headerLabels.append(["Model Name", "Covariates", "Est. Defects"] + percentNames)
        headerLabels.append(["Model Name", "Covariates", "Est. Budget"] + percentNames)
        return headerLabels


class SideMenu4(QVBoxLayout):
    """Side menu for tab 4.

    Attributes:
        allocation1Button: QPushButton object, begins effort allocation when
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
    runAllocation1Signal = pyqtSignal(list)  # starts allocation 1 computation
    runAllocation2Signal = pyqtSignal(list)  # starts allocation 2 computation

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
        modelsGroup = QGroupBox("Select Models for Allocation")
        modelsGroup.setLayout(self._setupModelsGroup())
        allocation1Group = self._setupAllocation1Group("Allocation 1")
        allocation2Group = self._setupAllocation2Group("Allocation 2")

        self.addWidget(modelsGroup, 10)
        self.addWidget(allocation1Group)
        self.addWidget(allocation2Group)

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


    def _setupAllocation1Group(self, label):
        group = QGroupBox(label)
        groupLayout = QVBoxLayout()

        # Effort allocation to discover maximum number of faults within given budget 'B'
        description = QLabel("Maximize defect discovery within\nbudget.")
        groupLayout.addWidget(description)
        verticalSpacer = QSpacerItem(20, 20, QSizePolicy.Fixed, QSizePolicy.Expanding)
        groupLayout.addItem(verticalSpacer)
        groupLayout.addWidget(QLabel("Enter budget"))

        self.budgetSpinBox = QDoubleSpinBox()
        self.budgetSpinBox.setRange(0.1, 999999.0)
        self.budgetSpinBox.setValue(20)
        groupLayout.addWidget(self.budgetSpinBox)

        self.allocation1Button = self._setupAllocationButton("Run Allocation 1", self._button1Pressed)
        groupLayout.addWidget(self.allocation1Button)
        group.setLayout(groupLayout)

        return group

    def _setupAllocation2Group(self, label):
        group = QGroupBox(label)
        groupLayout = QVBoxLayout()
        # Effort allocation to expose 'k' number of additional faults with the smallest budget possible
        description = QLabel("Minimum budget (B) to discover\nspecified additonal defects")
        groupLayout.addWidget(description)

        verticalSpacer = QSpacerItem(20, 20, QSizePolicy.Fixed, QSizePolicy.Preferred)
        groupLayout.addItem(verticalSpacer)

        groupLayout.addWidget(QLabel("Enter number of additional defects"))
        self.failureSpinBox = QSpinBox()
        self.failureSpinBox.setRange(1, 999999)
        groupLayout.addWidget(self.failureSpinBox)

        self.allocation2Button = self._setupAllocationButton("Run Allocation 2", self._button2Pressed)
        groupLayout.addWidget(self.allocation2Button)
        group.setLayout(groupLayout)

        return group

    def _setupAllocationButton(self, label, slot):
        """Creates the button that begins effort allocation."""
        button = QPushButton(label)
        button.setEnabled(False)    # begins disabled since no model has been run yet
        button.clicked.connect(slot)

        return button

    def _button1Pressed(self):
        self._emitRunAllocationSignal(1)

    def _button2Pressed(self):
        self._emitRunAllocationSignal(2)

    def _emitRunAllocationSignal(self, allocation_type):
        """Emits signal that effort allocation with model/metric combiations.

        Method called when Run Allocation button is pressed. The emitted signal
        (runAllocationSignal) contains the list of combinations to run the
        allocation on. The signal is only emitted if at least one combination
        is selected.
        """
        selectedCombinationNames = [item.text().split(". ", 1)[1] for item in self.modelListWidget.selectedItems()]
        if selectedCombinationNames:
            log.info("Selected for Allocation: %s", selectedCombinationNames)

            if allocation_type == 1:
                self.runAllocation1Signal.emit(selectedCombinationNames)
            else:
                self.runAllocation2Signal.emit(selectedCombinationNames)

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
