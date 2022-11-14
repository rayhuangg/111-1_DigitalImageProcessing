from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog

import cv2, os, copy, math, cmath, time
import numpy as np
from matplotlib import pyplot as plt

from UI import Ui_MainWindow

class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()


    # TODO
    def setup_control(self):
        self.ui.tabWidget.setCurrentIndex(0) # 設定從第一分頁開始顯示
        self.ui.button_file.clicked.connect(self.open_file)
        self.ui.button_file_2.clicked.connect(self.open_file)
        self.ui.button_file_3.clicked.connect(self.open_file)
        self.ui.button_file_4.clicked.connect(self.open_file)
        self.ui.button_fft.clicked.connect(self.part1_color_model)
        self.ui.button_go.clicked.connect(self.part1_select_filter)

        self.ui.horizontalSlider_cutoff.valueChanged.connect(self.slider_show)
        self.ui.horizontalSlider_c.valueChanged.connect(self.slider_show)
        self.ui.horizontalSlider_d0.valueChanged.connect(self.slider_show)
        self.ui.horizontalSlider_gh.valueChanged.connect(self.slider_show)
        self.ui.horizontalSlider_gl.valueChanged.connect(self.slider_show)
        self.ui.horizontalSlider_c.valueChanged.connect(self.homomorphic_filter)
        self.ui.horizontalSlider_d0.valueChanged.connect(self.homomorphic_filter)
        self.ui.horizontalSlider_gh.valueChanged.connect(self.homomorphic_filter)
        self.ui.horizontalSlider_gl.valueChanged.connect(self.homomorphic_filter)
        self.ui.comboBox_blur.currentTextChanged.connect(self.part4)


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


    def get_qimg(self, img, width=200, height=150):
        if type(img) == str: # 路徑的話就做imread,否則直接使用
            img = cv2.imread(img,    0) # gray
        elif type(img) == np.ndarray:
             img = img

        img = self.img_resize(img, width=width, height=height)

        # 記得對彩色圖及黑白圖片的QImage.Format、bytesPerline設定
        # self.qimg = QImage(self.img, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        if len(img.shape) == 3: # color img
            qimg = QImage(img, img.shape[1], img.shape[0], img.shape[1]*3, QImage.Format_RGB888).rgbSwapped()
        elif len(img.shape) == 2: # binary img
            qimg = QImage(img, img.shape[1], img.shape[0], img.shape[1], QImage.Format_Grayscale8).rgbSwapped()

        return qimg


    # get resize image
    def img_resize(self, img, width=200, height=150):
        if len(img.shape) == 3: # color img
            out = np.zeros((height, width, 3), np.uint8)
        elif len(img.shape) == 2: # binary img
            out = np.zeros((height, width), np.uint8)

        h, w = img.shape[0], img.shape[1]
        h2 = height / h
        w2 = width / w
        for i in range(height):
            for j in range(width):
                x = int(i / h2)
                y = int(j / w2)
                if len(img.shape) == 3: # color img
                    out[i, j, :] = img[x, y, :]
                elif len(img.shape) == 2: # binary img
                    out[i, j] = img[x, y]

        return out


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
        # if self.img_name is None:
        #     return
        self.img_name = self.img_name.split("/")[-1] # 取檔案路徑最後一個分割，即為檔名
        self.raw_img = cv2.imread(self.img_name, 0) # gray

        qimg = self.get_qimg(self.raw_img)

        # 判斷要顯示在哪個分頁上
        if self.ui.tabWidget.currentIndex() == 0:
            self.ui.label_img_1.setPixmap(QPixmap.fromImage(qimg))
        elif self.ui.tabWidget.currentIndex() == 1:
            self.ui.label_img_3.setPixmap(QPixmap.fromImage(qimg))
        elif self.ui.tabWidget.currentIndex() == 2:
            self.ui.label_img_5.setPixmap(QPixmap.fromImage(qimg))
        elif self.ui.tabWidget.currentIndex() == 3:
            qimg = self.get_qimg(self.raw_img, width=150, height=150)
            self.ui.label_img_8.setPixmap(QPixmap.fromImage(qimg))


    # 顯示sliderbar數值
    def slider_show(self):
        self.ui.label_cutoff.setText(str(self.ui.horizontalSlider_cutoff.value()))
        self.ui.label_gh.setText(str(self.ui.horizontalSlider_gh.value()))
        self.ui.label_gl.setText(str(self.ui.horizontalSlider_gl.value()))
        self.ui.label_c.setText(str(self.ui.horizontalSlider_c.value()))
        self.ui.label_d0.setText(str(self.ui.horizontalSlider_d0.value()))


    def grayscale(self, img):
        B = img[:,:,0]
        G = img[:,:,1]
        R = img[:,:,2]
        return np.array((R/3 + G/3 + B/3), dtype=np.uint8)


    def get_split_channel(self, img):
        return img[:,:,0], img[:,:,2], img[:,:,2]


    def part1_color_model(self):

        if (self.ui.comboBox_color_model.currentText()) == "CMY":
            self.idal_filter(type="low")
        elif (self.ui.comboBox_color_model.currentText()) == "HSI":
            self.idal_filter(type="high")
        elif (self.ui.comboBox_color_model.currentText()) == "XYZ":
            self.butterworth_filter(type="low")
        elif (self.ui.comboBox_color_model.currentText()) == "Lab":
            self.butterworth_filter(type="high")
        elif (self.ui.comboBox_color_model.currentText()) == "YUV":
            self.gaussian_filter(type="low")


        dft_shift, magnitude_spectrum = self.fft_cv2(self.raw_img)
        img_back = self.ifft_cv2(dft_shift)

        qimg = self.get_qimg(magnitude_spectrum)
        self.ui.label_img_2.setPixmap(QPixmap.fromImage(qimg))
        qimg = self.get_qimg(img_back)
        self.ui.label_img_7.setPixmap(QPixmap.fromImage(qimg))


    def rgb2cmy(self):
        img = self.raw_img
        b, g, r = self.get_split_channel(img)
        # normalization [0,1]
        r = r / 255.0
        g = g / 255.0
        b = b / 255.0
        # tranform to cmy
        c = 1 - r
        m = 1 - g
        y = 1 - b
        result = cv2.merge((c, m, y))


    def rgb2hsi(self):
        img = self.raw_img
        bgr = np.float32(img)/255 # convert to float type

        # Separate color channels
        b, g, r = self.get_split_channel(np.float32(img) / 255)
        # Separate color channels
        blue = bgr[:,:,0]
        green = bgr[:,:,1]
        red = bgr[:,:,2]

        # Calculate Intensity
        def calc_intensity(red, blue, green):
            return np.divide(blue + green + red, 3)

        # Calculate Saturation
        def calc_saturation(red, blue, green):
            minimum = np.minimum(np.minimum(red, green), blue)
            saturation = 1 - (3 / (red + green + blue + 0.001) * minimum)

            return saturation

        # Calculate Hue
        def calc_hue(red, blue, green):
            hue = np.copy(red)

            for i in range(0, blue.shape[0]):
                for j in range(0, blue.shape[1]):
                    hue[i][j] = 0.5 * ((red[i][j] - green[i][j]) + (red[i][j] - blue[i][j])) / \
                                math.sqrt((red[i][j] - green[i][j])**2 +
                                        ((red[i][j] - blue[i][j]) * (green[i][j] - blue[i][j])))
                    hue[i][j] = math.acos(hue[i][j])

                    if blue[i][j] <= green[i][j]:
                        hue[i][j] = hue[i][j]
                    else:
                        hue[i][j] = ((360 * math.pi) / 180.0) - hue[i][j]

            return hue

        # Merge channels into picture and return image
        hsi = cv2.merge((calc_hue(r, b, g), calc_saturation(r, b, g), calc_intensity(r, b, g)))
        # hsi = cv2.merge((calc_hue(red, blue, green), calc_saturation(red, blue, green), calc_intensity(red, blue, green)))


    # https://www.jianshu.com/p/c34a313e12eb
    def rgb2xyz(self):
        img = self.raw_img
        m = np.array([[0.412453, 0.357580, 0.180423],
                      [0.212671, 0.715160, 0.072169],
                      [0.019334, 0.119193, 0.950227]])

        b, g, r = self.get_split_channel(img)
        rgb = np.array([r, g, b])
        # rgb = rgb / 255.0
        # RGB = np.array([gamma(c) for c in rgb])
        XYZ = np.dot(m, rgb.T) / 255
        # XYZ = XYZ / 255.0
        result = (XYZ[0] / 0.95047, XYZ[1] / 1.0, XYZ[2] / 1.08883)

        plt.imshow(XYZ)
        plt.show()