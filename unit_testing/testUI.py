import unittest
import sys

from PyQt5.QtWidgets import QApplication  # for UI
import main
from ui.mainWindow import MainWindow

app = QApplication(sys.argv)

class TestUI(unittest.TestCase):

    def setUp(self):
        self.form = MainWindow()

    def test_demensions(self):
        self.assertEqual(self.form.title, "Covariate Tool")
        self.assertEqual(self.form.left, 100)
        self.assertEqual(self.form.top,100)
        self.assertEqual(self.form.width,1080)
        self.assertEqual(self.form.height,720)
        self.assertEqual(self.form.minWidth, 800)
        self.assertEqual(self.form.minHeight,600)




if __name__ == '__main__':
    unittest.main()
