import matplotlib.pyplot as plt
import matplotlib.ticker as ticker  # to force integers for bar chart x axis values
from PyQt5.QtCore import QSettings

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
            # can only have "post" parameter if using a step function
            plotMethod(x, y, self.style, markerSize=self.markerSize, where="post")  # ax.step()
            # plotMethod(x, y, self.style, markerSize=self.markerSize)  # ax.step()
        elif self.plotType == "bar":
            # ax.set_xticks(x)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            plotMethod(x, y, color='skyblue')    # ax.bar()
        else:
            plotMethod(x, y, self.style, markerSize=self.markerSize)    # ax.plot()
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
        plotMethod(x, y, self.style, markerSize=self.markerSize, label=label)
        return ax