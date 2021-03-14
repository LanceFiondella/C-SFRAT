import matplotlib.pyplot as plt
import matplotlib.ticker as ticker  # to force integers for bar chart x axis values
from PyQt5.QtCore import QSettings
from scipy.stats import norm
import numpy as np


class PlotSettings:
    def __init__(self):
        self._style = '-o'
        self._plotType = "step"
        self.markerSize = 3
        self.linewidth = 2
        self.color = "black"

    @property
    def style(self):
        return self._style

    @style.setter
    def style(self, style):
        self._style = style

    @property
    def plotType(self):
        return self._plotType

    @plotType.setter
    def plotType(self, plotType):
        self._plotType = plotType

    def generatePlot(self, ax, x, y, title="None", xLabel="X", yLabel="Y"):
        ax = self.setupPlot(ax, title=title, xLabel=xLabel, yLabel=yLabel)
        plotMethod = getattr(ax, self.plotType)     # equivalent to ax.plotType, depends on what plot type is
        if self.plotType == "step":
            x = x.to_numpy()
            y = y.to_numpy()

            # can only have "post" parameter if using a step function
            plotMethod(x, y, self.style, linewidth=self.linewidth, color=self.color, markersize=self.markerSize, where="post")  # ax.step()
        elif self.plotType == "bar":
            # ax.set_xticks(x)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))  
            plotMethod(x, y, color='darkgrey')  # ax.bar()
        else:
            # add point at (0, 0) if not there
            x = x.to_numpy()
            y = y.to_numpy()
            plotMethod(x, y, self.style, markersize=self.markerSize)    # ax.plot()
        return ax

    def setupPlot(self, ax, title="None", xLabel="X", yLabel="Y"):
        ax.clear()
        ax.grid(True)
        ax.set_title(title)
        ax.set_xlabel(xLabel)
        ax.set_ylabel(yLabel)
        return ax

    def addLine(self, ax, x, y, label="None"):
        plotMethod = getattr(ax, self.plotType)

        if self.plotType == "step":
            plotMethod(x, y, self.style, markersize=self.markerSize, where="post", label=label)
        else:
            plotMethod(x, y, self.style, markersize=self.markerSize, label=label)
        return ax

    def addZeroPoint(self, x, y):
        x = np.concatenate((np.zeros(1), x))
        y = np.concatenate((np.zeros(1), y))
        return x, y

    @staticmethod
    def predictionPlot(ax):
        pass
