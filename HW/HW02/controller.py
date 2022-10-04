from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog

import cv2, os, copy
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


    def getslidervalue(self):
        self.value = self.ui.horizontalSlider.value()
        pass


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
                self.gray_A[i,n] = (int(self.R[i,n]) + int(self.G[i,n]) + int(self.B[i,n])) // 3.0
                self.gray_B[i,n] = int(0.299 * int(self.R[i,n]) + 0.587 * int(self.G[i,n]) + 0.114 * int(self.B[i,n]))

        self.diff = np.zeros_like(self.G)
        for i in range(self.img.shape[0]):
            for n in range(self.img.shape[1]):
                if (int(self.gray_A[i,n]) - int(self.gray_B[i,n])) < 0:
                    self.diff[i,n] = 0
                else:
                    self.diff[i,n] = int(self.gray_A[i,n]) - int(self.gray_B[i,n])

        # TODO 參考https://codeantenna.com/a/sZw8f2E1Ie 精簡化顯示照片
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
        pass

    def Q2_5_shrink(self):
        pass

    def Q2_6_brightness_and_contrast(self):
        self.img = cv2.imread(self.img_name)

        # Get slider value
        self.brightness = self.ui.horizontalSlider_brightness.value()
        self.ui.label_brightness.setText(str(self.brightness))
        self.contrast = self.ui.horizontalSlider_contrast.value()
        self.ui.label_contrast.setText(str(self.contrast))

        # Adjust brightness
        # ref https://steam.oxxostudio.tw/category/python/ai/opencv-adjust.html
        self.brightness_img = self.img * 1.0 + self.brightness # 此處沒有*1.0就會出現奇怪顏色，包括*1也會，不確定原因
        self.brightness_img = np.uint8(np.clip(self.brightness_img, 0, 255)) # 避免溢位

        # Adjust contrast
        self.contrast_img = self.img * (self.contrast/127 + 1) - self.contrast
        self.contrast_img = np.uint8(np.clip(self.contrast_img, 0, 255)) # 避免溢位

        self.qimg = self.get_qimg(self.brightness_img)
        self.ui.label_img_7.setPixmap(QPixmap.fromImage(self.qimg))
        self.qimg = self.get_qimg(self.contrast_img)
        self.ui.label_img_8.setPixmap(QPixmap.fromImage(self.qimg))


        self.plot_histogram(self.brightness_img)
        self.qimg = self.get_qimg('Matplotlib.jpg')
        self.ui.label_hist_7.setPixmap(QPixmap.fromImage(self.qimg))
        os.remove('Matplotlib.jpg')
        self.plot_histogram(self.contrast_img)
        self.qimg = self.get_qimg('Matplotlib.jpg')
        self.ui.label_hist_8.setPixmap(QPixmap.fromImage(self.qimg))
        os.remove('Matplotlib.jpg')
