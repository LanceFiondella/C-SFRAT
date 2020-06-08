from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QGridLayout, \
                            QTableWidget, QTableWidgetItem, QAbstractScrollArea, \
                            QSpinBox, QSpacerItem, QSizePolicy, QHeaderView
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QFont

# Local imports
from core.comparison import Comparison

class Tab3(QWidget):
    """Contains all widgets displayed on tab 3.

    Attributes:
        sideMenu: SideMenu object holding tab 3 widgets and their signals.
        table: QTableWidget that contains the goodness-of-fit measures for each
            calculated model/metric combination.
        font: QFont object that is formatted bold. Used to set text bold for
            cells containing the highest ranked combinations, according to the
            weighting of each measure.
    """

    def __init__(self):
        """Initializes tab 3 UI elements."""
        super().__init__()
        self._setupTab3()

    def addResultsToTable(self, results):
        """Perfoms comparison calculations, adds goodness of fit results to table.

        Args:
            results: A dict containing the model objects as values, indexed by
                the name of the model/metric combination. The model objects
                contain the goodness of fit measures as properties.
        """
        # numResults = len(results)
        self.table.setSortingEnabled(False) # disable sorting while editing contents
        self.table.clear()
        self.table.setHorizontalHeaderLabels(["Model Name", "Covariates", "Log-Likelihood", "AIC", "BIC",
                                              "SSE", "Model ranking (no weights)", "Model ranking (user-specified weights)"])
                                              #"Weighted selection (mean)", "Weighted selection (median)"])
        self.table.setRowCount(len(results))    # set row count to include all model results, 
                                                # even if not converged
        i = 0   # number of converged models

        self.sideMenu.comparison.goodnessOfFit(results, self.sideMenu)

        for key, model in results.items():
            if model.converged:
                self.table.setItem(i, 0, QTableWidgetItem(model.shortName))
                self.table.setItem(i, 1, QTableWidgetItem(model.metricString))
                self.table.setItem(i, 2, QTableWidgetItem("{0:.3f}".format(model.llfVal)))
                self.table.setItem(i, 3, QTableWidgetItem("{0:.3f}".format(model.aicVal)))
                self.table.setItem(i, 4, QTableWidgetItem("{0:.3f}".format(model.bicVal)))
                self.table.setItem(i, 5, QTableWidgetItem("{0:.3f}".format(model.sseVal)))
                self.table.setItem(i, 6, QTableWidgetItem("{0:.3f}".format(self.sideMenu.comparison.meanOutUniform[i])))
                self.table.setItem(i, 7, QTableWidgetItem("{0:.3f}".format(self.sideMenu.comparison.meanOut[i])))
                i += 1
        self.table.setRowCount(i)   # set row count to only include converged models
        self.table.resizeColumnsToContents()    # resize column width after table is edited
        self.table.setSortingEnabled(True)      # re-enable sorting after table is edited

        self.table.item(self.sideMenu.comparison.bestMeanUniform, 6).setFont(self.font)
        self.table.item(self.sideMenu.comparison.bestMean, 7).setFont(self.font)

    def _setupTab3(self):
        """Creates tab 3 widgets and adds them to layout."""
        mainLayout = QHBoxLayout()       # main layout
        self.sideMenu = SideMenu3()
        self.table = self._setupTable()
        self.font = QFont() # allows table cells to be bold
        self.font.setBold(True)
        mainLayout.addLayout(self.sideMenu, 15)
        mainLayout.addWidget(self.table, 85)
        self.setLayout(mainLayout)

    def _setupTable(self):
        """Creates table widget with proper headers.

        Returns:
            A QTableWidget with specified column headers.
        """
        table = QTableWidget()
        table.setEditTriggers(QTableWidget.NoEditTriggers)     # make cells unable to be edited
        table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
                                                                    # column width fit to contents
        table.setRowCount(1)
        columnLabels = ["Model Name", "Covariates", "Log-Likelihood", "AIC", "BIC",
                        "SSE", "Model ranking (no weights)", "Model ranking (user-specified weights)"]
        table.setColumnCount(len(columnLabels))
        table.setHorizontalHeaderLabels(columnLabels)
        table.move(0,0)

        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)

        return table


class SideMenu3(QGridLayout):
    """ Side menu for tab 3.
    
    Attributes:
        comparison: Comparison object that performs the calculations to
            determine which combination best fits the data.
        llfSpinBox: QSpinBox object, specifies the weighting used for the
            log-likelihood function in the comparison.
        aicSpinBox: QSpinBox object, specifies the weighting used for the
            Akaike information criterion in the comparison.
        bicSpinBox: QSpinBox object, specifies the weighting used for the
            Bayesian information criterion in the comparison.
        sseSpinBox: QSpinBox object, specifies the weighting used for the sum
            of squares error in the comparison.
        spinBoxChangedSignal: pyqtSignal, emits when any of the spin boxes for
            goodness-of-fit comparison weighting are changed.
    """

    # signals
    spinBoxChangedSignal = pyqtSignal()

    def __init__(self):
        """Initializes tab 3 side menu UI elements."""
        super().__init__()
        self._setupSideMenu()
        self.comparison = Comparison()

    def _setupSideMenu(self):
        """Creates side menu widgets and adds them to the layout."""
        self._createLabel("Metric", 0, 0)
        self._createLabel("weights (0-10)", 0, 1)
        self._createLabel("LLF", 1, 0)
        self._createLabel("AIC", 2, 0)
        self._createLabel("BIC", 3, 0)
        self._createLabel("SSE", 4, 0)
        self.llfSpinBox = self._createSpinBox(0, 10, 1, 1)
        self.aicSpinBox = self._createSpinBox(0, 10, 2, 1)
        self.bicSpinBox = self._createSpinBox(0, 10, 3, 1)
        self.sseSpinBox = self._createSpinBox(0, 10, 4, 1)

        # vertical spacer at bottom of layout, keeps labels/spinboxes together at top of window
        vspacer = QSpacerItem(20, 40, QSizePolicy.Maximum, QSizePolicy.Expanding)
        self.addItem(vspacer, 5, 0, 1, -1)
        self.setColumnStretch(1, 1)

    def _createLabel(self, text, row, col):
        """Creates a text label and adds it to the side menu layout.

        Args:
            text: The string of text the label displays.
            row: The row of the QGroupBox widget to add the label to.
            col: The column of the QGroupBox widget to add the label to.
        """
        label = QLabel(text)
        label.setSizePolicy(QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum))
        self.addWidget(label, row, col)

    def _createSpinBox(self, minVal, maxVal, row, col):
        """Creates a QSpinBox and adds it to the side menu layout.

        Current weighting values are allowed to be between 0 and 10.

        Args:
            minVal: The minimum value allowed by the spinbox (int).
            maxVal: The maximum value allowed by the spinbox (int).
            row: The row of the QGroupBox widget to add the spinbox to.
            col: The column of the QGroupBox widget to add the spinbox to.
        Returns:
            A created QSpinBox object with specified parameters.
        """
        spinBox = QSpinBox()
        spinBox.setRange(minVal, maxVal)
        spinBox.setValue(1)  # give equal weighting of 1 by default
        spinBox.valueChanged.connect(self._emitSpinBoxChangedSignal)
        self.addWidget(spinBox, row, col)
        return spinBox

    def _emitSpinBoxChangedSignal(self):
        """Emits signal if any goodness-of-fit spin box is changed."""
        self.spinBoxChangedSignal.emit()
