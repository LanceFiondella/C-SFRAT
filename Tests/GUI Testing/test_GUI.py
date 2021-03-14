import sys
import unittest
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt


### Changes the curretnt working directory so the imports Work
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from ui.mainWindow import MainWindow
from core.dataClass import Data
import pandas as pd
import pyautogui
import time





app = QApplication(sys.argv)

class Covariatetest(unittest.TestCase):
    '''Testing the Covariate-Tool GUI'''
    def setUp(self):
        '''Creating the GUI and Load Data'''
        self.form = MainWindow()
        self.form.show()
        self.form.data.importFile("covariate_data.xlsx")
        #self.form.data.importFile("ds1.csv")
        self.form.dataLoaded = True
        self.form.importFileSignal.emit()



    def test_defaults(self):
        ''' Test the GUI in it's default state (Dimesions) '''

        self.assertEqual(self.form.windowTitle(),"C-SFRAT")
        self.assertEqual(self.form.geometry(), QtCore.QRect(100, 100, 1280, 960))
        self.assertEqual(self.form.minimumSize(),QtCore.QSize(1000,800))

    def test_file(self):
        ''' Test File Menu '''

        pyautogui.press('esc')
        self.form.fileOpened()
        pyautogui.press('esc')
        self.form.exportTable2()
        pyautogui.press('esc')
        self.form.exportTable3()

    def test_view(self):
        ''' Test View Menu '''

        self.form.setPointsView()
        self.form.setLineView()
        self.form.setLineAndPointsView()



if __name__ == "__main__":
    unittest.main()
    app.exec_()