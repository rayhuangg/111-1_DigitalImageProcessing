from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog

import cv2, os
import numpy as np
from matplotlib import pyplot as plt

# import matplotlib
# matplotlib.use('QT5Agg')
# from matplotlib.figure import Figure
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


from UI import Ui_MainWindow

class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        # TODO
        # self.ui.file_button1.clicked.connect(self.open_file)
        self.ui.button_file.clicked.connect(self.display_img)
        self.img_path = 'lena.jpg'
        self.display_img(path=self.img_path, objectName="label_img")
        self.plot_histogram(self.img)



    def display_img(self, path, objectName):
        self.img = cv2.imread(path)
        self.img = cv2.resize(self.img, (200,150))
        # print("type==========================", type(self.img))
        height, width, channel = self.img.shape
        bytesPerline = 3 * width
        self.qimg = QImage(self.img, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()

        # choose what label is gonna to show img.
        if objectName == "label_img":
            self.ui.label_img.setPixmap(QPixmap.fromImage(self.qimg))
        elif objectName == "label_hist":
            self.ui.label_hist.setPixmap(QPixmap.fromImage(self.qimg))


    def open_file(self):
        self.filename, filetype = QFileDialog.getOpenFileName(self,
                  "Open file",
                  "./")   # start path
        print(self.filename, filetype)
        self.ui.show_file_path1.setText(self.filename) # show file path

    # 繪製圖像直方圖
    def plot_histogram(self, img) :
        # if type(img) == str:
        #     img = cv2.imread('img')
        hist = np.zeros(256)
        # 計算各數值數量並儲存
        for i in np.unique(img):
            hist[i] = np.bincount(img.flatten())[i]
        # 利用長條圖繪出顯示
        x_axis = np.arange(256)
        plt.figure(figsize=(4,3))
        plt.bar(x_axis, hist)
        plt.title("Histogram")

        # show histogram
        plt.savefig("Matplotlib.jpg")
        # mat = cv2.imread("Matplotlib.jpg")
        # mat =
        self.display_img(path="Matplotlib.jpg", objectName="label_hist")
        os.remove("Matplotlib.jpg")