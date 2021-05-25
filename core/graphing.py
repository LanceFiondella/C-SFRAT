from PyQt5 import QtGui
from PyQt5 import QtCore

import pyqtgraph as pg
#import logging as log


class PlotWidget(pg.PlotWidget):
    """
    """

    # static variables
    lineStyle = None

    def __init__(self):
        """Initializes plot widget"""
        super().__init__()

        self.color = (255, 255, 255)
        self.setBackground(self.color)
        # self.showGrid(x=True, y=True)

        self.legendMVF = self.createLegend()
        self.legendIntensity = self.createLegend()

        # store pen used for vertical line so we don't need to make it every time
        self.penVerticalLine = pg.mkPen((255, 0, 0), width=2, style=QtCore.Qt.DashLine)

        self.plotColor = PlotColor()

        # PlotItem.plot()
        self.mvfPlotDataItem = None
        self.intensityPlotDataItem = None

        # contains PlotDataItem
        self.mvfLines = {}
        self.intensityLines = {}

        # names of model combinations currently displayed
        self.currentLines = []

        # styles
        self.lineStyle = "both"     # points, line, or both
        self.plotStyle = "smooth"   # smooth or step plot

        # self.verticalLine = None
        self.lastXpoint = 0

        self.createVerticalLine()

    def createLegend(self):
        pen = pg.mkPen(color=(0, 0, 0), width=1.0)
        brush = pg.mkBrush(color=(255, 255, 255))
        legend = pg.LegendItem(offset=(60, 10), verSpacing=-0.5, pen=pen, brush=brush, frame=True)#, colCount=2)
        legend.setLabelTextColor((0, 0, 0))    # black text
        # legend.setParentItem(plotItem)

        return legend

    def createPlots(self, x, y_mvf, y_intensity):
        """
        Called when importing data
        """
        self.legendMVF.clear()
        self.legendIntensity.clear()

        # get value of last element in pandas series
        self.lastXpoint = x.iloc[-1]

        self.createMvfPlot(x, y_mvf)
        self.createIntensityPlot(x, y_intensity)

        self.addVerticalLine()

    def createMvfPlot(self, x, y):
        # should only be called when new data is loaded
        self.mvfPlotItem = pg.PlotItem()
        self.mvfPlotItem.showGrid(x=True, y=True)
        self.mvfPlotItem.setLabel("bottom", "Intervals")
        self.mvfPlotItem.setLabel("left", "Cumulative failures")
        pen = pg.mkPen(color=(0, 0, 0), width=5)
        self.mvfPlotDataItem = pg.PlotDataItem(x, y, pen=pen, stepMode='right')
        self.mvfPlotItem.addItem(self.mvfPlotDataItem)

        self.legendMVF.setParentItem(self.mvfPlotItem)
        self.legendMVF.addItem(self.mvfPlotDataItem, "Imported data")

        # # get value of last element in pandas series
        # self.lastXpoint = x.iloc[-1]
        # print(self.lastXpoint)

    def createIntensityPlot(self, x, y):
        # should only be called when new data is loaded
        self.intensityPlotItem = pg.PlotItem()
        self.intensityPlotItem.showGrid(x=True, y=True)
        self.mvfPlotItem.setLabel("bottom", "Intervals")
        self.mvfPlotItem.setLabel("left", "Failures")
        self.intensityPlotDataItem = pg.BarGraphItem(x=x, height=y, width=0.8, brush=(200, 200, 200))
        self.intensityPlotItem.addItem(self.intensityPlotDataItem)

        self.legendIntensity.setParentItem(self.intensityPlotItem)
        self.legendIntensity.addItem(self.intensityPlotDataItem, "Imported data")

    def addVerticalLine(self):
        # can't remove item before it is added
        # only an issue the first time model fitting is run
        try:
            self.mvfPlotItem.removeItem(self.verticalLine1)
            self.intensityPlotItem.removeItem(self.verticalLine2)
        except:
            pass

        pen = pg.mkPen((255, 0, 0), width=2, style=QtCore.Qt.DashLine)

        self.verticalLine1 = pg.InfiniteLine(pos=self.lastXpoint, angle=90, pen=pen)
        self.verticalLine2 = pg.InfiniteLine(pos=self.lastXpoint, angle=90, pen=pen)

        self.verticalLine1.setPos(self.lastXpoint)
        self.verticalLine2.setPos(self.lastXpoint)

        self.mvfPlotItem.addItem(self.verticalLine1)
        self.intensityPlotItem.addItem(self.verticalLine2)

    def createVerticalLine(self):
        pen = pg.mkPen((255, 0, 0), width=0, style=QtCore.Qt.DashLine)

        self.verticalLine1 = pg.InfiniteLine(pos=self.lastXpoint, angle=90, pen=pen)
        self.verticalLine2 = pg.InfiniteLine(pos=self.lastXpoint, angle=90, pen=pen)

    def changePlotType(self, plotViewIndex):
        self.clear()
        # MVF
        if plotViewIndex == 0:
            self.plotItem = self.mvfPlotItem
            self.setCentralItem(self.plotItem)
        # intensity
        elif plotViewIndex == 1:
            self.plotItem = self.intensityPlotItem
            self.setCentralItem(self.plotItem)

    def createLines(self, results):
        """
        called when estimation is complete
        creates line objects for all models, for mvf and intensity plots
        """

        self.plotColor.index = 0

        # clear plots (by removing fitted lines)
        # needed when model fitting is run while previous results are displayed on tab 2 plots
        self.removeLines(self.currentLines)

        # clear dictionaries so they do not continue to grow as different
        # model combinations are run
        self.mvfLines.clear()
        self.intensityLines.clear()
        self.currentLines.clear()

        for key, model in results.items():
            color = self.plotColor.nextColor()
            pen = pg.mkPen(color, width=3)
            symbolBrush = pg.mkBrush(color)
            self.mvfLines[key] = pg.PlotDataItem(model.t, model.mvf_array, pen=pen)
            self.mvfLines[key].setSymbolPen(pen)
            self.mvfLines[key].setSymbolBrush(symbolBrush)
            self.mvfLines[key].setSymbol('o')

            self.intensityLines[key] = pg.PlotDataItem(model.t, model.intensityList, pen=pen)
            self.intensityLines[key].setSymbolPen(pen)
            self.intensityLines[key].setSymbolBrush(symbolBrush)
            self.intensityLines[key].setSymbol('o')

            # check for line style
            if self.lineStyle == "points":
                self.setPointsView()

            elif self.lineStyle == "line":
                self.setLineView()

            elif self.lineStyle == "both":
                self.setLineAndPointsView()

            # check for plot style
            if self.plotStyle == "smooth":
                self.setSmoothPlot()
            elif self.plotStyle == "step":
                self.setStepPlot()


    def updateLines(self, newLines):
        # more lines than currently shown means we need to add lines
        if len(newLines) > len(self.currentLines):
            lines = [x for x in newLines if x not in self.currentLines]
            self.addLines(lines)
        # fewer lines than currently shown means we need to remove lines
        elif len(newLines) < len(self.currentLines):
            lines = [x for x in self.currentLines if x not in newLines]
            self.removeLines(lines)

        # resize window size to ensure all lines are visible
        self.resizePlot()
        self.currentLines = newLines   # lines changed, so current lines updated

    def addLines(self, lines):
        for line in lines:
            self.mvfPlotItem.addItem(self.mvfLines[line])
            self.intensityPlotItem.addItem(self.intensityLines[line])

            self.legendMVF.addItem(self.mvfLines[line], line)
            self.legendIntensity.addItem(self.intensityLines[line], line)
            # self.legend.setColumnCount(2)

    def removeLines(self, lines):
        for line in lines:
            self.mvfPlotItem.removeItem(self.mvfLines[line])
            self.intensityPlotItem.removeItem(self.intensityLines[line])

            self.legendMVF.removeItem(self.mvfLines[line])
            self.legendIntensity.removeItem(self.intensityLines[line])

    def setPointsView(self):
        for line in self.mvfLines:
            self.mvfLines[line].setSymbolSize(4)
            self.mvfLines[line].setPen((0, 0, 0, 0))    # transparent (4th value)

            self.intensityLines[line].setSymbolSize(4)
            self.intensityLines[line].setPen((0, 0, 0, 0))  # transparent (4th value)

        self.lineStyle = "points"

    def setLineView(self):
        self.plotColor.index = 0
        for line in self.mvfLines:
            color = self.plotColor.nextColor()
            self.mvfLines[line].setSymbolSize(0)
            self.mvfLines[line].setPen(color, width=3)

            self.intensityLines[line].setSymbolSize(0)
            self.intensityLines[line].setPen(color, width=3)

        self.lineStyle = "line"

    def setLineAndPointsView(self):
        self.plotColor.index = 0
        for line in self.mvfLines:
            color = self.plotColor.nextColor()
            self.mvfLines[line].setSymbolSize(4)
            self.mvfLines[line].setPen(color, width=3)

            self.intensityLines[line].setSymbolSize(4)
            self.intensityLines[line].setPen(color, width=3)

        self.lineStyle = "both"

    def setSmoothPlot(self):
        for line in self.mvfLines:
            self.mvfLines[line].opts['stepMode'] = None
            self.mvfLines[line].updateItems()

            self.intensityLines[line].opts['stepMode'] = None
            self.intensityLines[line].updateItems()

        self.plotStyle = "smooth"

    def setStepPlot(self):
        for line in self.mvfLines:
            self.mvfLines[line].opts['stepMode'] = 'right'
            self.mvfLines[line].updateItems()

            self.intensityLines[line].opts['stepMode'] = 'right'
            self.intensityLines[line].updateItems()

        self.plotStyle = "step"

    def subsetPlots(self, x, y_mvf, y_intensity):
        ## new max interval, getData returns subset
        # mvf plot
        self.mvfPlotDataItem.setData(x, y_mvf)

        # intensity plot
        self.intensityPlotDataItem.setOpts(x=x, height=y_intensity)

        self.lastXpoint = x.iloc[-1]

        self.addVerticalLine()

    def updateLineMVF(self, model, x, y):
        self.mvfLines[model].setData(x, y)
        self.resizePlot()

    def updateLineIntensity(self, model, x, y):
        self.intensityLines[model].setData(x, y)
        self.resizePlot()

    def resizePlot(self):
        self.mvfPlotItem.autoRange()
        self.intensityPlotItem.autoRange()


class PlotColor:
    # blue, orange, green, red, purple, brown, pink, grey, olive, cyan
    colors = [
        QtGui.QColor(31, 119, 191),     # blue
        QtGui.QColor(255, 127, 14),     # orange
        QtGui.QColor(44, 160, 44),      # green
        QtGui.QColor(214, 39, 40),      # red
        QtGui.QColor(148, 103, 189),    # purple
        QtGui.QColor(140, 92, 75),      # brown
        QtGui.QColor(227, 119, 194),    # pink
        QtGui.QColor(127, 127, 127),    # grey
        QtGui.QColor(188, 189, 34),     # olive
        QtGui.QColor(23, 190, 207)      # cyan
    ]

    def __init__(self):
        self._index = 0

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, i):
        self._index = i % len(PlotColor.colors)

    def nextColor(self):
        color = PlotColor.colors[self.index]
        self.index += 1
        return color
