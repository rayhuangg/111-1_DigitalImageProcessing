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

        self.ui.button_file.clicked.connect(self.open_and_return_filename)
        # self.ui.button_file.clicked.connect(self.display_img)
        self.ui.button_2_2_grayscale.clicked.connect(self.Q2_2_grayscale)
        self.ui.button_2_3_histogram.clicked.connect(self.Q2_3_histogram)
        self.ui.button_2_4_threshold.clicked.connect(self.Q2_4_threshold)
        self.ui.button_2_5_enlarge.clicked.connect(self.Q2_5_enlarge)
        self.ui.button_2_5_shrink.clicked.connect(self.Q2_5_shrink)
        self.ui.button_2_6_brightness_an_contrast.clicked.connect(self.Q2_6_brightness_and_contrast)

        # self.img_name = 'lena.jpg'
        # self.display_img(path=self.img_name, objectName="label_img")
        # self.plot_histogram(self.img)



    def display_img(self, img_name, objectName):
        if type(img_name) == str: # 路徑的話就做imread,否則直接使用
            self.img = cv2.imread(img_name)
        elif type(img_name) == np.ndarray:
             self.img = img_name

        self.img = cv2.resize(self.img, (200,150))
        height, width = self.img.shape[0], self.img.shape[1]
        bytesPerline = 3 * width
        self.qimg = QImage(self.img, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()

        # choose what label is gonna to show img.
        if objectName == "label_img":
            self.ui.label_img.setPixmap(QPixmap.fromImage(self.qimg))
        elif objectName == "label_hist":
            self.ui.label_hist.setPixmap(QPixmap.fromImage(self.qimg))


    def open_and_return_filename(self):
        self.img_name, _ = QFileDialog.getOpenFileName(self,
                  "Open file",
                  "./")   # start path
        print(self.img_name)
        self.img_name = self.img_name.split("/")[-1] # 取檔案路徑最後一個分割，即為檔名
        self.ui.show_file_path1.setText(self.img_name) # show file path
        self.display_img(img_name=self.img_name, objectName="label_img")
        self.plot_histogram(self.img, objectName="label_hist")

    # plot histogram
    def plot_histogram(self, img, objectName):
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
        self.display_img(img_name="Matplotlib.jpg", objectName=objectName)
        os.remove("Matplotlib.jpg")

    def Q2_2_grayscale(self):
        img = cv2.imread(self.img_name)

        # split color channel
        R = img[:,:,0]
        G = img[:,:,1]
        B = img[:,:,2]
        gray_A = np.zeros_like(R) # 複製大小
        gray_B = np.zeros_like(R)

        for i in range(img.shape[0]):
            for n in range(img.shape[1]):
                gray_A[i,n] = (R[i,n]) + int(G[i,n]) + int(B[i,n]) // 3
                gray_B[i,n] = int(0.299 * int(R[i,n]) + 0.587 * int(G[i,n]) + 0.114 * int(B[i,n]))

        # TODO 參考https://codeantenna.com/a/sZw8f2E1Ie精簡化顯示照片

        # self.display_img(gray_A, "label_img_2")
        QImage(gray_A, gray_A[1], gray_A[0], gray_A[1] * 3, QImage.Format_RGB888).rgbSwapped()  # 此处如果不加*3，就会发生倾斜
        self.plot_histogram(gray_A, 'label_hist_2')
        self.display_img(gray_B, "label_img_3")
        self.plot_histogram(gray_B, 'label_hist_3')



    def Q2_3_histogram(self):
        pass

    def Q2_4_threshold(self):
        pass

    def Q2_5_enlarge(self):
        pass

    def Q2_5_shrink(self):
        pass

    def Q2_6_brightness_and_contrast(self):
        pass