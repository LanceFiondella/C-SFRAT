import matplotlib.pyplot as plt
import matplotlib.ticker as ticker  # to force integers for bar chart x axis values
from PyQt5.QtCore import QSettings
from scipy.stats import norm
import numpy as np

from core.dataClass import PandasModel

class PlotSettings:
    def __init__(self):
        self._style = '-o'
        self._plotType = "step"
        self.markerSize = 3

    @property
    def style(self):
        return self._style

    @style.setter
    def style(self, style):
        self._style = style

    @property
    def plotType(self):
        return self._plotType

    @ plotType.setter
    def plotType(self, plotType):
        self._plotType = plotType

    def generatePlot(self, ax, x, y, title="None", xLabel="X", yLabel="Y"):
        ax = self.setupPlot(ax, title=title, xLabel=xLabel, yLabel=yLabel)
        plotMethod = getattr(ax, self.plotType)     # equivalent to ax.plotType, depends on what plot type is
        if self.plotType == "step":
            # add point at (0, 0) if not there
            x = x.to_numpy()
            y = y.to_numpy()
            # print(x[0])
            if x[0] != 0:
                x, y = self.addZeroPoint(x, y)

            # can only have "post" parameter if using a step function
            plotMethod(x, y, self.style, markerSize=self.markerSize, where="post")  # ax.step()
            # plotMethod(x, y, self.style, markerSize=self.markerSize)  # ax.step()
        elif self.plotType == "bar":
            # ax.set_xticks(x)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            plotMethod(x, y, color='skyblue')    # ax.bar()
        else:
            # add point at (0, 0) if not there
            x = x.to_numpy()
            y = y.to_numpy()
            # print(x[0])
            if x[0] != 0:
                x, y = self.addZeroPoint(x, y)

            plotMethod(x, y, self.style, markerSize=self.markerSize)    # ax.plot()
        return ax

    def setupPlot(self, ax, title="None", xLabel="X", yLabel="Y"):
        ax.clear()
        ax.grid(True)
        ax.set_title(title)
        ax.set_xlabel(xLabel)
        ax.set_ylabel(yLabel)
        return ax

    @staticmethod
    def addLaplaceLines(ax, confidence):
        # values taken from SFRAT R code
        ax.axhline(y=norm.ppf(0.1), color='silver', linestyle='dotted')
        ax.axhline(y=norm.ppf(0.05), color='silver', linestyle='dotted')
        ax.axhline(y=norm.ppf(0.01), color='silver', linestyle='dotted')
        ax.axhline(y=norm.ppf(0.001), color='silver', linestyle='dotted')
        ax.axhline(y=norm.ppf(0.0000001), color='silver', linestyle='dotted')
        ax.axhline(y=norm.ppf(0.0000000001), color='silver', linestyle='dotted')
        ax.axhline(y=norm.ppf(1.0 - confidence), color='red', linestyle='-')    # specified confidence level

    @staticmethod
    def updateConfidenceLine(ax, confidence):
        # ax.lines[-1].remove()
        # ax.axhline(y=norm.ppf(1.0 - confidence), color='red', linestyle='-')

        # running arithmetic average only has one line, don't want to change that one
        if len(ax.lines) > 1: 
            ax.lines[-1].set_ydata(norm.ppf(1.0 - confidence))

    def addLine(self, ax, x, y, label="None"):
        plotMethod = getattr(ax, self.plotType)

        # add point at (0, 0) if not there
        if int(x[0]) != 0:
            x, y = self.addZeroPoint(x, y)
        if self.plotType == "step":
            plotMethod(x, y, self.style, markerSize=self.markerSize, where="post", label=label)
        else:
            plotMethod(x, y, self.style, markerSize=self.markerSize, label=label)
        return ax

    def addZeroPoint(self, x, y):
        # print(type(x))
        # print(np.zeros(1))
        # print(x)
        x = np.concatenate((np.zeros(1), x))
        y = np.concatenate((np.zeros(1), y))
        return x, y

    @staticmethod
    def predictionPlot(ax):
        pass