# from PyQt5 import QtWidgets, QtGui, QtCore
# import pyqtgraph as pg
# import sys
# import os

# class MainWindow(QtWidgets.QMainWindow):

#     def __init__(self, *args, **kwargs):
#         super(MainWindow, self).__init__(*args, **kwargs)

#         self.graphWidget = pg.PlotWidget()
#         self.setCentralWidget(self.graphWidget)

#         hour = [1,2,3,4,5,6,7,8,9,10]
#         temperature = [30,32,34,32,33,31,29,32,35,45]

#         pen = pg.mkPen(color=(0, 0, 0), width=5)

#         color = self.palette().color(QtGui.QPalette.Window)  # Get the default window background,
#         self.graphWidget.setBackground(color)

#         self.graphWidget.showGrid(x=True, y=True)

#         self.graphWidget.setLabel("left", "Temperature (°C)")
#         self.graphWidget.setLabel("bottom", "Hour (H)")


#         # plot data: x, y values
#         line = self.graphWidget.plot(hour, temperature, pen=pen)
#         pen = pg.mkPen(color=(255, 0, 0), width=5, style=QtCore.Qt.DashLine)
#         self.graphWidget.setXRange(min(hour), max(hour))#, padding=0)
#         self.graphWidget.setYRange(min(temperature), max(temperature))#, padding=0)
#         static_line = self.graphWidget.plot([3, 3], [min(temperature)*-2, max(temperature)*2], pen=pen)
#         static_line.clear()

#         # static_line = pg.InfiniteLine(pos=3, angle=90)

#         print(static_line)


# def main():
#     app = QtWidgets.QApplication(sys.argv)
#     main = MainWindow()
#     main.show()
#     sys.exit(app.exec_())


# if __name__ == '__main__':
#     main()







###########################################







## https://codeloop.org/how-to-plot-bargraph-in-pyqtgraph/

# import sys
# from PyQt5.QtWidgets import QApplication
# import pyqtgraph as pg
# import numpy as np
 
# app = QApplication(sys.argv)
 
# win = pg.plot()
 
 
# x = np.arange(0, 2*3.14, 0.1)
# print(x)
# y1 = np.sin(x)
# y2 = 1.1 * np.sin(x+1)
# y3 = 1.2 * np.sin(x+2)
 
# bg1 = pg.BarGraphItem(x=x, height=y1, width=0.1, brush='b')
# # bg2 = pg.BarGraphItem(x=x+0.33, height=y2, width=0.3, brush='g')
# # bg3 = pg.BarGraphItem(x=x+0.66, height=y3, width=0.3, brush='b')
 
# win.addItem(bg1)
# # win.addItem(bg2)
# # win.addItem(bg3)
 
 
 
# # # Final example shows how to handle mouse clicks:
# # class BarGraph(pg.BarGraphItem):
# #     def mouseClickEvent(self, event):
# #         print("clicked")
 
 
 
 
# # bg = BarGraph(x=x, y=y1*0.3+2, height=0.4+y1*0.2, width=0.8)
# # win.addItem(bg)
 
 
 
 
# status = app.exec_()
# sys.exit(status)




##########################


## https://stackoverflow.com/questions/54803737/pyqt5-pyqtgraph-adding-removing-curves-in-a-single-plot
# from PyQt5.QtGui import* 
# from PyQt5.QtCore import*
# from PyQt5.QtWidgets import*
# import pyqtgraph as pg
# import numpy as np
# import sys

# class MyWidget(QWidget):
#     def __init__(self, parent=None):
#         super(MyWidget, self).__init__(parent)
#         self.win = pg.GraphicsWindow()
#         self.p = []
#         self.c = []
#         for i in range(3):
#             self.p.append(self.win.addPlot(row=i, col=0))
#             for j in range(2):
#                 self.c.append(self.p[-1].plot(np.random.rand(100), pen=3*i+j))
#         self.update()
#         self.del_curve()
#         self.add_curve()

#     def update(self): # update a curve
#         self.c[3].setData(np.random.rand(100)*10)

#     def del_curve(self): # remove a curve
#         self.c[5].clear()

#     def add_curve(self): # add a curve
#         self.c.append(self.p[2].plot(np.random.rand(100)))

# def startWindow():
#     app = QApplication(sys.argv)
#     mw = MyWidget()
#     app.exec_()

# if __name__ == '__main__':
#     startWindow()