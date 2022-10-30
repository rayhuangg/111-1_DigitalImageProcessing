# from sympy import *
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog

import cv2, os, copy, math, time
import numpy as np
from matplotlib import pyplot as plt

from UI import Ui_MainWindow

class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        # TODO
        self.ui.button_file.clicked.connect(self.open_file)
        self.ui.button_fft.clicked.connect(self.fft)


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
        self.img = cv2.imread(self.img_name)

        qimg = self.get_qimg(self.img_name)
        self.ui.label_img_1.setPixmap(QPixmap.fromImage(qimg))


    # def forward(self):
        # '''
        # Performs a forward pass of the conv layer using the given input.
        # - input is a 2d numpy array
        # '''
        # input = self.img
        # h, w = input.shape
        # # output = np.zeros((h - 2, w - 2, self.num_filters))
        # output = np.zeros((h-2, w-2))
        # self.filter = self.set_filter()

        # for im_region, i, j in self.iterate_regions(input):
        #     output[i, j] = np.sum(im_region * self.filter)
        #     output = np.clip(np.uint8(output), 0, 255)

        # self.qimg = self.get_qimg(self.img_name)
        # self.ui.label_img_2.setPixmap(QPixmap.fromImage(self.qimg))


    def fft(self):
        src = cv2.imread(self.img_name, 0) # gray

        start = time.process_time()
        # 將影像進行float轉換才能進行dft
        result = cv2.dft(np.float32(src), flags=cv2.DFT_COMPLEX_OUTPUT)
        # 將spectrum平移到中心
        dft_shift = np.fft.fftshift(result)
        # 將spectrum複數轉換為 0-255 區間
        result1 = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

        qimg = self.get_qimg(result1)
        self.ui.label_img_2.setPixmap(QPixmap.fromImage(qimg))

        # plt.subplot(121), plt.imshow(src, 'gray'), plt.title('oreginal')
        # plt.axis('off')
        # plt.subplot(122), plt.imshow(result1, 'gray'), plt.title('FFT')
        # plt.axis('off')
        # plt.show()
        end = time.process_time()
        print(f"Process time: {(end-start):0.10f} s")

        # return result1

