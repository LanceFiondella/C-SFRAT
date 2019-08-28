# PyQt5 imports for UI elements
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTabWidget, QTableView, \
                            QProgressBar, QLabel
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtCore import Qt

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
        self.figure = self.plotWidget.figure
        self.addTab(self.plotWidget, 'Plot')

    def setupTableTab(self):
        self.tableWidget = QTableView()
        self.addTab(self.tableWidget, 'Table')

class ComputeWidget(QWidget):
    results = pyqtSignal(dict)

    # predict points?
    def __init__(self, modelsToRun, metricNames, data, parent=None):
        super(ComputeWidget, self).__init__(parent)
        layout = QVBoxLayout(self)
        
        # set fixed window size (width, height)
        self.setFixedSize(250, 150)

        self.progressBar = QProgressBar(self)
        self.progressBar.setRange(0, len(modelsToRun))
        self.label = QLabel()
        self.label.setText("Computing results...\nModels completed: {0}".format(0))

        layout.addWidget(self.label)
        layout.addWidget(self.progressBar)
        layout.setAlignment(Qt.AlignVCenter)
        self.setWindowTitle("Processing...")

        self.computeTask = TaskThread(modelsToRun, metricNames, data)
        self.computeTask.modelFinished.connect(self.modelFinished)
        self.computeTask.taskFinished.connect(self.onFinished)
        self.computeTask.start()

        self.show()


    def modelFinished(self):
        self.progressBar.setValue(self.progressBar.value() + 1)
        self.label.setText("Computing results...\nModels completed: {0}".format(self.progressBar.value() + 1))

    def onFinished(self, result):
        self.results.emit(result)
        self.close()


class TaskThread(QThread):
    modelFinished = pyqtSignal()
    taskFinished = pyqtSignal(dict)

    # predict points?
    def __init__(self, modelsToRun, metricNames, data):
        super().__init__()
        self.modelsToRun = modelsToRun
        self.metricNames = metricNames
        self.data = data

    def run(self):
        result = {}
        for model in self.modelsToRun:
            m = model(data=self.data.getData(), metricNames=self.metricNames)
            # m.findParams(self.predictPoints)
            result[m.name] = m
            self.modelFinished.emit()
        self.taskFinished.emit(result)
