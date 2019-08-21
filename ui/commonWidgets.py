# PyQt5 imports for UI elements
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTabWidget, QTableView

# Matplotlib imports for graphs/plots
from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5
from matplotlib.backends.backend_qt5agg import FigureCanvas, \
                                    NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

class PlotWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setupPlot()

    def setupPlot(self):
        plotLayout = QVBoxLayout()
        self.figure = Figure(tight_layout={"pad": 2.0})
        self.plotFigure = FigureCanvas(self.figure)
        toolbar = NavigationToolbar(self.plotFigure, self)
        plotLayout.addWidget(self.plotFigure, 1)
        plotLayout.addWidget(toolbar)
        self.setLayout(plotLayout)

class PlotAndTable(QTabWidget):
    def __init__(self):
        super().__init__()
        self.setupPlotTab()
        self.setupTableTab()

    def setupPlotTab(self):
        # Creating plot widget
        self.plotWidget = PlotWidget()
        self.addTab(self.plotWidget, 'Plot')

    def setupTableTab(self):
        self.tableWidget = QTableView()
        self.addTab(self.tableWidget, 'Table')