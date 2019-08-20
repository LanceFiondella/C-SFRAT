from PyQt5 import QtWidgets, QtGui, QtCore
import sys
class Window(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        self.vlayout = QtWidgets.QVBoxLayout()
        self.pushButton_ok = QtWidgets.QPushButton("Press me", self)
        self.pushButton_ok.clicked.connect(self.addCheckbox)
        self.vlayout.addWidget(self.pushButton_ok)

        self.checkBox = QtWidgets.QCheckBox(self)
        self.vlayout.addWidget(self.checkBox)
        self.setLayout(self.vlayout)

    def addCheckbox(self):
        #checkBox = QtWidgets.QCheckBox()
        self.vlayout.addWidget(QtWidgets.QCheckBox()) 

application = QtWidgets.QApplication(sys.argv)
window = Window()
window.setWindowTitle('Dynamically adding checkboxes using a push button')
window.resize(250, 180)
window.show()
sys.exit(application.exec_())