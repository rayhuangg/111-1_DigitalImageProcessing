# from sympy import *
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import QtWidgets

import cv2, os, copy, math
import numpy as np
from matplotlib import pyplot as plt


class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        # TODO

        self.ui.button_file.clicked.connect(self.open_file)
        self.ui.button_filter.clicked.connect(self.forward)


    # plot histogram
    def plot_histogram(self, img):
        self.hist = np.zeros(256)
        # 計算各數值數量並儲存
        for i in np.unique(img):
            self.hist[i] = np.bincount(img.flatten())[i]

        self.x_axis = np.arange(256)
        plt.figure(figsize=(4,3))
        plt.bar(self.x_axis, self.hist)
        plt.title("Histogram")
        plt.savefig("Matplotlib.jpg")


    def get_qimg(self, img):
        if type(img) == str: # 路徑的話就做imread,否則直接使用
            self.img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        elif type(img) == np.ndarray:
             self.img = img
        # self.img = cv2.resize(self.img, (200,150))

        self.img = self.img_resize(self.img, width=200, height=150)

        # 記得對彩色圖及黑白圖片的QImage.Format、bytesPerline設定
        # self.qimg = QImage(self.img, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        if len(self.img.shape) == 3: # color img
            self.qimg = QImage(self.img, self.img.shape[1], self.img.shape[0], self.img.shape[1]*3, QImage.Format_RGB888).rgbSwapped()
        elif len(self.img.shape) == 2: # binary img
            self.qimg = QImage(self.img, self.img.shape[1], self.img.shape[0], self.img.shape[1], QImage.Format_Grayscale8).rgbSwapped()

        return self.qimg


    # get resize image
    def img_resize(self, img, width=200, height=150):
        out = np.zeros((height, width), np.uint8)
        h, w = img.shape
        h2 = height / h
        w2 = width / w
        for i in range(height):
            for j in range(width):
                x = int(i / h2)
                y = int(j / w2)
                out[i, j] = img[x, y]
        return out


    def set_filter(self, size=3):
        filter =  np.array([[1,1,1],
                            [1,1,1],
                            [1,1,1]]) / 9
        return filter


    def iterate_regions(self, image):
        h, w = image.shape
        # 將 im_region, i, j 存到迭代器中，讓後面遍歷
        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image[i:(i + 3), j:(j + 3)]
                yield im_region, i, j


    # zero padding
    def padding(self, data, n=2):
        # data = np.pad(data, ((n,n),(n,n)))
        return np.pad(data, ((n,n),(n,n)))


    def open_file(self):
        self.img_name, _ = QFileDialog.getOpenFileName(self,
                "Open file",
                "./")   # start path

        self.img_name = self.img_name.split("/")[-1] # 取檔案路徑最後一個分割，即為檔名
        self.img = cv2.imread(self.img_name, cv2.IMREAD_GRAYSCALE)

        self.qimg = self.get_qimg(self.img_name)
        self.ui.label_img_1.setPixmap(QPixmap.fromImage(self.qimg))


    def forward(self):
        '''
        Performs a forward pass of the conv layer using the given input.
        - input is a 2d numpy array
        '''
        input = self.img
        h, w = input.shape
        # output = np.zeros((h - 2, w - 2, self.num_filters))
        output = np.zeros((h-2, w-2))
        self.filter = self.set_filter()

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.sum(im_region * self.filter)
            output = np.clip(np.uint8(output), 0, 255)

        self.qimg = self.get_qimg(self.img_name)
        self.ui.label_img_2.setPixmap(QPixmap.fromImage(self.qimg))




class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label_img_1 = QtWidgets.QLabel(self.centralwidget)
        self.label_img_1.setGeometry(QtCore.QRect(20, 140, 200, 150))
        self.label_img_1.setAlignment(QtCore.Qt.AlignCenter)
        self.label_img_1.setObjectName("label_img_1")
        self.label_img_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_img_2.setGeometry(QtCore.QRect(330, 140, 200, 150))
        self.label_img_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_img_2.setObjectName("label_img_2")
        self.label_threshold_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_threshold_2.setGeometry(QtCore.QRect(10, 0, 101, 51))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(20)
        self.label_threshold_2.setFont(font)
        self.label_threshold_2.setTextFormat(QtCore.Qt.MarkdownText)
        self.label_threshold_2.setObjectName("label_threshold_2")
        self.button_file = QtWidgets.QPushButton(self.centralwidget)
        self.button_file.setGeometry(QtCore.QRect(10, 60, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.button_file.setFont(font)
        self.button_file.setObjectName("button_file")
        self.button_filter = QtWidgets.QPushButton(self.centralwidget)
        self.button_filter.setGeometry(QtCore.QRect(310, 60, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.button_filter.setFont(font)
        self.button_filter.setObjectName("button_filter")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_img_1.setText(_translate("MainWindow", "img"))
        self.label_img_2.setText(_translate("MainWindow", "result"))
        self.label_threshold_2.setText(_translate("MainWindow", "Part 2"))
        self.button_file.setText(_translate("MainWindow", "choose img"))
        self.button_filter.setText(_translate("MainWindow", "filter"))



if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow_controller()
    window.show()
    sys.exit(app.exec_())