# from sympy import *
import PyQt5
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog

import cv2, os, copy, math
import numpy as np
from matplotlib import pyplot as plt

import os
import qt5_applications
dirname = os.path.dirname(qt5_applications.__file__)
plugin_path = os.path.join(dirname, 'Qt', 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path



class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        # TODO

        self.ui.button_file.clicked.connect(self.Q2_1_open_file)
        self.ui.button_2_2_grayscale.clicked.connect(self.Q2_2_grayscale)
        self.ui.button_2_3_histogram.clicked.connect(self.Q2_3_histogram)
        self.ui.button_2_4_threshold.clicked.connect(self.Q2_4_threshold)
        self.ui.button_2_5_enlarge.clicked.connect(self.Q2_5_enlarge)
        self.ui.button_2_5_shrink.clicked.connect(self.Q2_5_shrink)
        self.ui.button_2_6_brightness.clicked.connect(self.Q2_6_brightness)
        self.ui.button_2_6_contrast.clicked.connect(self.Q2_6_contrast)


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
            self.img = cv2.imread(img)
        elif type(img) == np.ndarray:
             self.img = img
        self.img = cv2.resize(self.img, (200,150))

        # 記得對彩色圖及黑白圖片的QImage.Format、bytesPerline設定
        # self.qimg = QImage(self.img, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        if len(self.img.shape) == 3: # color img
            self.qimg = QImage(self.img, self.img.shape[1], self.img.shape[0], self.img.shape[1]*3, QImage.Format_RGB888).rgbSwapped()
        elif len(self.img.shape) == 2: # binary img
            self.qimg = QImage(self.img, self.img.shape[1], self.img.shape[0], self.img.shape[1], QImage.Format_Grayscale8).rgbSwapped()

        return self.qimg


    def Q2_1_open_file(self):
        self.img_name, _ = QFileDialog.getOpenFileName(self,
                "Open file",
                "./")   # start path

        self.img_name = self.img_name.split("/")[-1] # 取檔案路徑最後一個分割，即為檔名
        self.ui.show_file_path1.setText(self.img_name) # show file path

        self.qimg = self.get_qimg(self.img_name)
        self.ui.label_img.setPixmap(QPixmap.fromImage(self.qimg))

        self.plot_histogram(self.img)
        self.qimg = self.get_qimg('Matplotlib.jpg')
        self.ui.label_hist.setPixmap(QPixmap.fromImage(self.qimg))
        os.remove('Matplotlib.jpg')



    def Q2_2_grayscale(self):
        self.img = cv2.imread(self.img_name)

        # split color channel (opencv: BGR)
        self.B = self.img[:,:,0]
        self.G = self.img[:,:,1]
        self.R = self.img[:,:,2]
        # self.gray_A = np.zeros_like(self.R) # 複製圖像大小
        # self.gray_B = np.zeros_like(self.G)

        self.gray_A = np.array((self.R/3 + self.G/3 + self.B/3), dtype=np.uint8)
        self.gray_B = np.array((0.299*(self.R) + 0.587*(self.G) + 0.114*(self.B)), dtype=np.uint8)
        self.diff = np.clip((self.gray_A - self.gray_B), 0, 255)

        # 參考 https://codeantenna.com/a/sZw8f2E1Ie 精簡化顯示照片
        self.qimg = self.get_qimg(self.gray_A)
        self.ui.label_img_2.setPixmap(QPixmap.fromImage(self.qimg))

        self.qimg = self.get_qimg(self.gray_B)
        self.ui.label_img_3.setPixmap(QPixmap.fromImage(self.qimg))

        self.qimg = self.get_qimg(self.diff)
        self.ui.label_img_4.setPixmap(QPixmap.fromImage(self.qimg))


    def Q2_3_histogram(self):

        self.plot_histogram(self.gray_A)
        self.qimg = self.get_qimg('Matplotlib.jpg')
        self.ui.label_hist_2.setPixmap(QPixmap.fromImage(self.qimg))
        os.remove('Matplotlib.jpg')

        self.plot_histogram(self.gray_B)
        self.qimg = self.get_qimg('Matplotlib.jpg')
        self.ui.label_hist_3.setPixmap(QPixmap.fromImage(self.qimg))
        os.remove('Matplotlib.jpg')

        self.plot_histogram(self.diff)
        self.qimg = self.get_qimg('Matplotlib.jpg')
        self.ui.label_hist_4.setPixmap(QPixmap.fromImage(self.qimg))
        os.remove('Matplotlib.jpg')


    def Q2_4_threshold(self):
        # Get slider value
        self.threshold = self.ui.horizontalSlider_threshold.value()
        self.ui.label_threshold.setText(str(self.threshold))

        # Threshold
        self.binary_image = copy.deepcopy(self.gray_A) # 複製gray image
        for i in range(self.gray_A.shape[0]):
            for n in range(self.gray_A.shape[1]):
                if self.binary_image[i,n] > self.threshold:
                    self.binary_image[i,n] = 255

        self.qimg = self.get_qimg(self.binary_image)
        self.ui.label_img_5.setPixmap(QPixmap.fromImage(self.qimg))


    def Q2_5_enlarge(self):
        self.gray = copy.deepcopy(self.gray_A)

        self.enlarge = np.zeros_like(self.gray)

        ## Calculate weight by distance
        for x in range(self.gray.shape[0]):
            for y in range(self.gray.shape[1]):
                a = self.gray[x,y,0] - math.floor(self.gray[x,y,0])
                b = self.gray[x,y,1] - math.floor(self.gray[x,y,1])

                self.enlarge[x,y] = a*b*self.gray[x+1,y+1] \
                                    + (1-a)*b*self.gray[x,y+1] \
                                    + a*(1-b)*self.gray[x+1,y] \
                                    + (1-a)*(1-b)*self.gray[x,y]

        ## Bilinear interpolation
        # for x in range(1, self.gray.shape[0]-1):
        #     for y in range(1, self.gray.shape[1]-1):
        #         a, b, c, d = symbols('a, b, c, d', communtative=True)
        #         self.solved_value = solve((a*(x-1) + b*y + c*(x-1)*y + d - self.gray[x-1, y],
        #                             a*x + b*(y+1) + c*x*(y+1) + d - self.gray[x, y+1],
        #                             a*x + b*(y+1) + c*x*(y+1) + d - self.gray[x, y-1],
        #                             a*(x+1) + b*y + c*(x+1)*y + d - self.gray[x+1, y]),
        #                             (a,b,c,d))



    def Q2_5_shrink(self):
        pass

    def Q2_6_brightness(self):
        self.img = cv2.imread(self.img_name)

        # Get slider value
        self.brightness = self.ui.horizontalSlider_brightness.value()
        self.ui.label_brightness.setText(str(self.brightness))

        # Adjust brightness
        # ref https://steam.oxxostudio.tw/category/python/ai/opencv-adjust.html
        self.brightness_img = self.img * 1.0 + self.brightness # 此處沒有*1.0就會出現奇怪顏色，包括*1也會，不確定原因
        self.brightness_img = np.uint8(np.clip(self.brightness_img, 0, 255)) # 避免溢位

        self.qimg = self.get_qimg(self.brightness_img)
        self.ui.label_img_7.setPixmap(QPixmap.fromImage(self.qimg))


        self.plot_histogram(self.brightness_img)
        self.qimg = self.get_qimg('Matplotlib.jpg')
        self.ui.label_hist_7.setPixmap(QPixmap.fromImage(self.qimg))
        os.remove('Matplotlib.jpg')


    def Q2_6_contrast(self):
        self.img = cv2.imread(self.img_name)

        # Get slider value
        self.contrast = self.ui.horizontalSlider_contrast.value()
        self.ui.label_contrast.setText(str(self.contrast))

        # Adjust contrast
        # ref https://steam.oxxostudio.tw/category/python/ai/opencv-adjust.html
        self.contrast_img = self.img * (self.contrast/127 + 1) - self.contrast
        self.contrast_img = np.uint8(np.clip(self.contrast_img, 0, 255)) # 避免溢位

        self.qimg = self.get_qimg(self.contrast_img)
        self.ui.label_img_8.setPixmap(QPixmap.fromImage(self.qimg))

        self.plot_histogram(self.contrast_img)
        self.qimg = self.get_qimg('Matplotlib.jpg')
        self.ui.label_hist_8.setPixmap(QPixmap.fromImage(self.qimg))
        os.remove('Matplotlib.jpg')

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1920, 1080)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.button_file = QtWidgets.QPushButton(self.centralwidget)
        self.button_file.setGeometry(QtCore.QRect(70, 50, 111, 31))
        self.button_file.setObjectName("button_file")
        self.show_file_path1 = QtWidgets.QTextEdit(self.centralwidget)
        self.show_file_path1.setGeometry(QtCore.QRect(240, 40, 201, 41))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.show_file_path1.setFont(font)
        self.show_file_path1.setObjectName("show_file_path1")
        self.label_img = QtWidgets.QLabel(self.centralwidget)
        self.label_img.setGeometry(QtCore.QRect(80, 110, 200, 150))
        self.label_img.setObjectName("label_img")
        self.label_hist = QtWidgets.QLabel(self.centralwidget)
        self.label_hist.setGeometry(QtCore.QRect(310, 110, 200, 150))
        self.label_hist.setObjectName("label_hist")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(10, 270, 531, 20))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(220, 340, 141, 31))
        self.textBrowser.setObjectName("textBrowser")
        self.label_hist_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_hist_2.setGeometry(QtCore.QRect(310, 400, 200, 150))
        self.label_hist_2.setObjectName("label_hist_2")
        self.label_img_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_img_2.setGeometry(QtCore.QRect(80, 400, 200, 150))
        self.label_img_2.setObjectName("label_img_2")
        self.label_hist_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_hist_3.setGeometry(QtCore.QRect(310, 620, 200, 150))
        self.label_hist_3.setObjectName("label_hist_3")
        self.label_img_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_img_3.setGeometry(QtCore.QRect(80, 620, 200, 150))
        self.label_img_3.setObjectName("label_img_3")
        self.label_hist_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_hist_4.setGeometry(QtCore.QRect(310, 830, 200, 150))
        self.label_hist_4.setObjectName("label_hist_4")
        self.label_img_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_img_4.setGeometry(QtCore.QRect(80, 830, 200, 150))
        self.label_img_4.setObjectName("label_img_4")
        self.textBrowser_2 = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_2.setGeometry(QtCore.QRect(220, 570, 231, 31))
        self.textBrowser_2.setObjectName("textBrowser_2")
        self.textBrowser_3 = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_3.setGeometry(QtCore.QRect(220, 780, 141, 31))
        self.textBrowser_3.setObjectName("textBrowser_3")
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setGeometry(QtCore.QRect(540, 0, 31, 1011))
        self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.textBrowser_4 = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_4.setGeometry(QtCore.QRect(70, 10, 61, 31))
        self.textBrowser_4.setObjectName("textBrowser_4")
        self.label_img_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_img_5.setGeometry(QtCore.QRect(680, 130, 200, 150))
        self.label_img_5.setObjectName("label_img_5")
        self.button_2_2_grayscale = QtWidgets.QPushButton(self.centralwidget)
        self.button_2_2_grayscale.setGeometry(QtCore.QRect(70, 290, 111, 41))
        self.button_2_2_grayscale.setObjectName("button_2_2_grayscale")
        self.button_2_3_histogram = QtWidgets.QPushButton(self.centralwidget)
        self.button_2_3_histogram.setGeometry(QtCore.QRect(70, 340, 111, 41))
        self.button_2_3_histogram.setObjectName("button_2_3_histogram")
        self.button_2_4_threshold = QtWidgets.QPushButton(self.centralwidget)
        self.button_2_4_threshold.setGeometry(QtCore.QRect(630, 40, 111, 41))
        self.button_2_4_threshold.setObjectName("button_2_4_threshold")
        self.button_2_5_enlarge = QtWidgets.QPushButton(self.centralwidget)
        self.button_2_5_enlarge.setGeometry(QtCore.QRect(640, 520, 111, 41))
        self.button_2_5_enlarge.setObjectName("button_2_5_enlarge")
        self.label_img_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_img_6.setGeometry(QtCore.QRect(680, 610, 200, 150))
        self.label_img_6.setObjectName("label_img_6")
        self.button_2_5_shrink = QtWidgets.QPushButton(self.centralwidget)
        self.button_2_5_shrink.setGeometry(QtCore.QRect(800, 520, 111, 41))
        self.button_2_5_shrink.setObjectName("button_2_5_shrink")
        self.line_3 = QtWidgets.QFrame(self.centralwidget)
        self.line_3.setGeometry(QtCore.QRect(570, 400, 491, 20))
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.button_2_6_brightness = QtWidgets.QPushButton(self.centralwidget)
        self.button_2_6_brightness.setGeometry(QtCore.QRect(1130, 30, 201, 41))
        self.button_2_6_brightness.setObjectName("button_2_6_brightness")
        self.label_img_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_img_7.setGeometry(QtCore.QRect(1140, 180, 200, 150))
        self.label_img_7.setObjectName("label_img_7")
        self.label_hist_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_hist_7.setGeometry(QtCore.QRect(1140, 340, 200, 150))
        self.label_hist_7.setObjectName("label_hist_7")
        self.textBrowser_5 = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_5.setGeometry(QtCore.QRect(1170, 90, 101, 41))
        self.textBrowser_5.setObjectName("textBrowser_5")
        self.textBrowser_6 = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_6.setGeometry(QtCore.QRect(1460, 90, 101, 41))
        self.textBrowser_6.setObjectName("textBrowser_6")
        self.label_hist_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_hist_8.setGeometry(QtCore.QRect(1430, 340, 200, 150))
        self.label_hist_8.setObjectName("label_hist_8")
        self.label_img_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_img_8.setGeometry(QtCore.QRect(1430, 180, 200, 150))
        self.label_img_8.setObjectName("label_img_8")
        self.horizontalSlider_threshold = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider_threshold.setGeometry(QtCore.QRect(770, 60, 160, 22))
        self.horizontalSlider_threshold.setMaximum(255)
        self.horizontalSlider_threshold.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_threshold.setObjectName("horizontalSlider_threshold")
        self.horizontalSlider_brightness = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider_brightness.setGeometry(QtCore.QRect(1140, 140, 160, 22))
        self.horizontalSlider_brightness.setMinimum(-255)
        self.horizontalSlider_brightness.setMaximum(255)
        self.horizontalSlider_brightness.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_brightness.setObjectName("horizontalSlider_brightness")
        self.horizontalSlider_contrast = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider_contrast.setGeometry(QtCore.QRect(1430, 140, 160, 22))
        self.horizontalSlider_contrast.setMinimum(-100)
        self.horizontalSlider_contrast.setMaximum(100)
        self.horizontalSlider_contrast.setProperty("value", 0)
        self.horizontalSlider_contrast.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_contrast.setObjectName("horizontalSlider_contrast")
        self.label_threshold = QtWidgets.QLabel(self.centralwidget)
        self.label_threshold.setGeometry(QtCore.QRect(820, 40, 47, 12))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_threshold.setFont(font)
        self.label_threshold.setObjectName("label_threshold")
        self.label_brightness = QtWidgets.QLabel(self.centralwidget)
        self.label_brightness.setGeometry(QtCore.QRect(1320, 140, 47, 12))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_brightness.setFont(font)
        self.label_brightness.setObjectName("label_brightness")
        self.label_contrast = QtWidgets.QLabel(self.centralwidget)
        self.label_contrast.setGeometry(QtCore.QRect(1600, 140, 47, 12))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_contrast.setFont(font)
        self.label_contrast.setObjectName("label_contrast")
        self.button_2_6_contrast = QtWidgets.QPushButton(self.centralwidget)
        self.button_2_6_contrast.setGeometry(QtCore.QRect(1410, 30, 201, 41))
        self.button_2_6_contrast.setObjectName("button_2_6_contrast")
        self.line_4 = QtWidgets.QFrame(self.centralwidget)
        self.line_4.setGeometry(QtCore.QRect(1060, 0, 31, 1041))
        self.line_4.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.line_5 = QtWidgets.QFrame(self.centralwidget)
        self.line_5.setGeometry(QtCore.QRect(1070, 510, 701, 20))
        self.line_5.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.button_2_7_histogram_equalazation = QtWidgets.QPushButton(self.centralwidget)
        self.button_2_7_histogram_equalazation.setGeometry(QtCore.QRect(1110, 540, 201, 41))
        self.button_2_7_histogram_equalazation.setObjectName("button_2_7_histogram_equalazation")
        self.label_threshold_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_threshold_2.setGeometry(QtCore.QRect(1180, 620, 47, 12))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_threshold_2.setFont(font)
        self.label_threshold_2.setObjectName("label_threshold_2")
        self.label_threshold_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_threshold_3.setGeometry(QtCore.QRect(1470, 620, 47, 12))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_threshold_3.setFont(font)
        self.label_threshold_3.setObjectName("label_threshold_3")
        self.label_img_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_img_10.setGeometry(QtCore.QRect(1410, 690, 200, 150))
        self.label_img_10.setObjectName("label_img_10")
        self.label_img_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_img_9.setGeometry(QtCore.QRect(1120, 690, 200, 150))
        self.label_img_9.setObjectName("label_img_9")
        self.label_hist_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_hist_9.setGeometry(QtCore.QRect(1120, 850, 200, 150))
        self.label_hist_9.setObjectName("label_hist_9")
        self.label_hist_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_hist_10.setGeometry(QtCore.QRect(1410, 850, 200, 150))
        self.label_hist_10.setObjectName("label_hist_10")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1920, 21))
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
        self.button_file.setText(_translate("MainWindow", "2-1 choose img"))
        self.show_file_path1.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'PMingLiU\'; font-size:8pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p>\n"
"<p align=\"center\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:11pt;\"><br /></p>\n"
"<p align=\"center\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:11pt;\"><br /></p></body></html>"))
        self.label_img.setText(_translate("MainWindow", "img"))
        self.label_hist.setText(_translate("MainWindow", "histogram"))
        self.textBrowser.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'PMingLiU\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">GRAY = (R+G+B)/3.0</p></body></html>"))
        self.label_hist_2.setText(_translate("MainWindow", "histogram"))
        self.label_img_2.setText(_translate("MainWindow", "img"))
        self.label_hist_3.setText(_translate("MainWindow", "histogram"))
        self.label_img_3.setText(_translate("MainWindow", "img"))
        self.label_hist_4.setText(_translate("MainWindow", "histogram"))
        self.label_img_4.setText(_translate("MainWindow", "img"))
        self.textBrowser_2.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'PMingLiU\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">GRAY = 0.299*R + 0.587*G + 0.114*B</p></body></html>"))
        self.textBrowser_3.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'PMingLiU\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">image subtraction</p></body></html>"))
        self.textBrowser_4.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'PMingLiU\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">2-1</p></body></html>"))
        self.label_img_5.setText(_translate("MainWindow", "img"))
        self.button_2_2_grayscale.setText(_translate("MainWindow", "2-2 grayscale"))
        self.button_2_3_histogram.setText(_translate("MainWindow", "2-3 histogram"))
        self.button_2_4_threshold.setText(_translate("MainWindow", "2-4 threshold"))
        self.button_2_5_enlarge.setText(_translate("MainWindow", "2-5 enlarge"))
        self.label_img_6.setText(_translate("MainWindow", "img"))
        self.button_2_5_shrink.setText(_translate("MainWindow", "2-5 shrink"))
        self.button_2_6_brightness.setText(_translate("MainWindow", "2-6 brightness"))
        self.label_img_7.setText(_translate("MainWindow", "img"))
        self.label_hist_7.setText(_translate("MainWindow", "histogram"))
        self.textBrowser_5.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'PMingLiU\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">brightness</p>\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">(-255~255)</p></body></html>"))
        self.textBrowser_6.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'PMingLiU\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">contrast</p>\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">(-100~100)</p></body></html>"))
        self.label_hist_8.setText(_translate("MainWindow", "histogram"))
        self.label_img_8.setText(_translate("MainWindow", "img"))
        self.label_threshold.setText(_translate("MainWindow", "value"))
        self.label_brightness.setText(_translate("MainWindow", "value"))
        self.label_contrast.setText(_translate("MainWindow", "value"))
        self.button_2_6_contrast.setText(_translate("MainWindow", "2-6 constrast"))
        self.button_2_7_histogram_equalazation.setText(_translate("MainWindow", "2-7 histogram equalization"))
        self.label_threshold_2.setText(_translate("MainWindow", "Before"))
        self.label_threshold_3.setText(_translate("MainWindow", "After"))
        self.label_img_10.setText(_translate("MainWindow", "img"))
        self.label_img_9.setText(_translate("MainWindow", "img"))
        self.label_hist_9.setText(_translate("MainWindow", "histogram"))
        self.label_hist_10.setText(_translate("MainWindow", "histogram"))


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow_controller()
    window.show()
    sys.exit(app.exec_())