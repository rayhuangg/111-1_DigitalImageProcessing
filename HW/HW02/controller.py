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

        self.ui.button_file.clicked.connect(self.Q2_1_open_file)
        self.ui.button_2_2_grayscale.clicked.connect(self.Q2_2_grayscale)
        self.ui.button_2_3_histogram.clicked.connect(self.Q2_3_histogram)
        self.ui.button_2_4_threshold.clicked.connect(self.Q2_4_threshold)
        self.ui.button_2_5_enlarge.clicked.connect(self.Q2_5_enlarge)
        self.ui.button_2_5_shrink.clicked.connect(self.Q2_5_shrink)
        self.ui.button_2_6_brightness_an_contrast.clicked.connect(self.Q2_6_brightness_and_contrast)

        # self.img_name = 'lena.jpg'
        # self.display_img(path=self.img_name, objectName="label_img")
        # self.plot_histogram(self.img)



    # def display_img(self, img_name, objectName):
        # if type(img_name) == str: # 路徑的話就做imread,否則直接使用
        #     self.img = cv2.imread(img_name)
        # elif type(img_name) == np.ndarray:
        #      self.img = img_name

        # self.img = cv2.resize(self.img, (200,150))
        # height, width = self.img.shape[0], self.img.shape[1]
        # bytesPerline = 3 * width
        # self.qimg = QImage(self.img, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()

        # # choose what label is gonna to show img.
        # if objectName == "label_img":
        #     self.ui.label_img.setPixmap(QPixmap.fromImage(self.qimg))
        # elif objectName == "label_hist":
        #     self.ui.label_hist.setPixmap(QPixmap.fromImage(self.qimg))


    # plot histogram
    def plot_histogram(self, img, objectName="1"):
        self.hist = np.zeros(256)
        # 計算各數值數量並儲存
        for i in np.unique(img):
            self.hist[i] = np.bincount(img.flatten())[i]

        self.x_axis = np.arange(256)
        plt.figure(figsize=(4,3))
        plt.bar(self.x_axis, self.hist)
        plt.title("Histogram")

        # show histogram
        plt.savefig("Matplotlib.jpg")
        # self.display_img(img_name="Matplotlib.jpg", objectName=objectName)
        # os.remove("Matplotlib.jpg")


    def get_qimg(self, img):
        if type(img) == str: # 路徑的話就做imread,否則直接使用
            self.img = cv2.imread(img)
        elif type(img) == np.ndarray:
             self.img = img
        self.img = cv2.resize(self.img, (200,150))

        # 記得對彩色圖及黑白圖片的QImage.Format、bytesPerline設定
        # self.qimg = QImage(self.img, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        if len(self.img.shape) == 3:
            self.qimg = QImage(self.img, self.img.shape[1], self.img.shape[0], self.img.shape[1]*3, QImage.Format_RGB888).rgbSwapped()
        elif len(self.img.shape) == 2:
            self.qimg = QImage(self.img, self.img.shape[1], self.img.shape[0], self.img.shape[1], QImage.Format_Grayscale8).rgbSwapped()

        return self.qimg



    def Q2_1_open_file(self):
        self.img_name, _ = QFileDialog.getOpenFileName(self,
                "Open file",
                "./")   # start path
        # print(self.img_name)
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
        self.gray_A = np.zeros_like(self.R) # 複製圖像大小
        self.gray_B = np.zeros_like(self.G)

        for i in range(self.img.shape[0]):
            for n in range(self.img.shape[1]):
                self.gray_A[i,n] = (int(self.R[i,n]) + int(self.G[i,n]) + int(self.B[i,n])) / 3.0
                self.gray_B[i,n] = int(0.299 * int(self.R[i,n]) + 0.587 * int(self.G[i,n]) + 0.114 * int(self.B[i,n]))

        print(self.img.shape)
        print(self.gray_A.shape)
        print(self.gray_B.shape)


        # TODO 參考https://codeantenna.com/a/sZw8f2E1Ie 精簡化顯示照片
        self.qimg1 = self.get_qimg(self.gray_A)
        self.ui.label_img_2.setPixmap(QPixmap.fromImage(self.qimg1))

        print(type(self.qimg1))

        self.qimg2 = self.get_qimg(self.gray_B)
        self.ui.label_img_3.setPixmap(QPixmap.fromImage(self.qimg2))

        # self.plot_histogram(self.gray_A, 'label_hist_2')
        # self.display_img(self.gray_B, "label_img_3")
        # self.plot_histogram(self.gray_B, 'label_hist_3')

### TODO check shape 位置


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