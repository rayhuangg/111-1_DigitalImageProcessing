from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog

import cv2
import numpy as np
import matplotlib
matplotlib.use('QT5Agg')
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


from UI import Ui_MainWindow

class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        # TODO
        self.ui.file_button1.clicked.connect(self.open_file)
        # self.ui.file_button2.clicked.connect(self.open_folder)

    def open_file(self):
        filename, filetype = QFileDialog.getOpenFileName(self,
                  "Open file",
                  "./")   # start path
        # print(filename, filetype)
        self.ui.show_file_path1.setText(filename)


class MyFigureCanvas(FigureCanvas):
    pass

