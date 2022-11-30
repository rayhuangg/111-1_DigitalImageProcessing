from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog
from PyQt5 import QtCore, QtGui, QtWidgets

import cv2, os, copy, math, cmath, time
import numpy as np
from matplotlib import pyplot as plt

from UI import Ui_MainWindow


class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()  # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

    # TODO
    def setup_control(self):
        self.ui.tabWidget.setCurrentIndex(0) # 設定從第一分頁開始顯示
        self.ui.button_file_1.clicked.connect(self.open_file)
        self.ui.button_file_2.clicked.connect(self.open_file)
        self.ui.button_file_3.clicked.connect(self.open_file)
        self.ui.button_trapezoidal.clicked.connect(self.part1_trapezoidal)
        self.ui.button_wavy.clicked.connect(self.part1_wavy)
        self.ui.button_circular.clicked.connect(self.part1_circular)

    # plot histogram
    def plot_histogram(self, img):
        self.hist = np.zeros(256)
        # 計算各數值數量並儲存
        for i in np.unique(img):
            self.hist[i] = np.bincount(img.flatten())[i]

        self.x_axis = np.arange(256)
        plt.figure(figsize=(4, 3))
        plt.bar(self.x_axis, self.hist)
        plt.title("Histogram")
        plt.savefig("Matplotlib.jpg")

    def get_qimg(self, img, width=200, height=150):
        # print(type(img), img)
        if type(img) == str:  # 路徑的話就做imread,否則直接使用
            img = cv2.imread(img, 0)  # gray
        elif type(img) == np.ndarray:
            img = img

        img = self.img_resize(img, width=width, height=height)

        # 記得對彩色圖及黑白圖片的QImage.Format、bytesPerline設定
        # self.qimg = QImage(self.img, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        if len(img.shape) == 3:  # color img
            qimg = QImage(
                img, img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888
            ).rgbSwapped()
        elif len(img.shape) == 2:  # binary img
            qimg = QImage(
                img, img.shape[1], img.shape[0], img.shape[1], QImage.Format_Grayscale8
            ).rgbSwapped()

        return qimg

    # get resize image
    def img_resize(self, img, width=200, height=150):
        if len(img.shape) == 3:  # color img
            out = np.zeros((height, width, 3), np.uint8)
        elif len(img.shape) == 2:  # binary img
            out = np.zeros((height, width), np.uint8)

        h, w = img.shape[0], img.shape[1]
        h2 = height / h
        w2 = width / w
        for i in range(height):
            for j in range(width):
                x = int(i / h2)
                y = int(j / w2)
                if len(img.shape) == 3:  # color img
                    out[i, j, :] = img[x, y, :]
                elif len(img.shape) == 2:  # binary img
                    out[i, j] = img[x, y]

        return out

    def iterate_regions(self, image):
        h, w = image.shape
        # 將 im_region, i, j 存到迭代器中，讓後面遍歷
        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image[i : (i + 3), j : (j + 3)]
                yield im_region, i, j

    # zero padding
    def padding(self, data, n=2):
        # data = np.pad(data, ((n,n),(n,n)))
        return np.pad(data, ((n, n), (n, n)))

    def open_file(self):
        self.img_name, _ = QFileDialog.getOpenFileName(
            self, "Open file", "./"
        )  # start path
        # if self.img_name is None:
        #     return
        # print(self.img_name, type(self.img_name))
        self.img_name = self.img_name.split("/")[-1]  # 取檔案路徑最後一個分割，即為檔名
        self.raw_img = cv2.imread(self.img_name, 0)  # gray
        # print(self.img_name, type(self.img_name))

        # 判斷要顯示在哪個分頁上
        if self.ui.tabWidget.currentIndex() == 0:
            qimg = self.get_qimg(self.raw_img, width=200, height=200)
            self.ui.label_img_1.setPixmap(QPixmap.fromImage(qimg))
        elif self.ui.tabWidget.currentIndex() == 1:
            qimg = self.get_qimg(self.raw_img, width=200, height=150)
            self.ui.label_img_3.setPixmap(QPixmap.fromImage(qimg))
        elif self.ui.tabWidget.currentIndex() == 2:
            # self.raw_img = cv2.imread('Part 3 Image/rects.bmp', 0)
            qimg = self.get_qimg(self.raw_img, width=200, height=200)
            self.ui.label_img_5.setPixmap(QPixmap.fromImage(qimg))
            self.part3_hough()

    # 顯示sliderbar數值
    def slider_show(self):
        self.ui.label_cutoff.setText(str(self.ui.horizontalSlider_cutoff.value()))
        self.ui.label_gh.setText(str(self.ui.horizontalSlider_gh.value()))
        self.ui.label_gl.setText(str(self.ui.horizontalSlider_gl.value()))
        self.ui.label_c.setText(str(self.ui.horizontalSlider_c.value()))
        self.ui.label_d0.setText(str(self.ui.horizontalSlider_d0.value()))

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

    def grayscale(self, img):
        B = img[:, :, 0]
        G = img[:, :, 1]
        R = img[:, :, 2]
        return np.array((R / 3 + G / 3 + B / 3), dtype=np.uint8)

    def fft_cv2(self, img):
        if len(img.shape) == 3:
            img = self.grayscale(img)

        # 將影像進行float轉換才能進行dft
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        # 將spectrum平移到中心
        dft_shift = np.fft.fftshift(dft)
        # 將spectrum複數空間轉換為 0-255 區間
        magnitude_spectrum = 20 * np.log(
            cv2.magnitude(x=dft_shift[:, :, 0], y=dft_shift[:, :, 1])
        )

        return dft_shift, magnitude_spectrum

    # def ifft_cv2(self, dft_shift):
    #     idft_shift = np.fft.ifftshift(dft_shift)
    #     img_back = cv2.idft(idft_shift, flags=cv2.DFT_SCALE )
    #     # img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    #     return img_back

    def ifft_cv2(self, dft_shift):
        # 反向平移
        idft_shift = np.fft.ifftshift(dft_shift)
        if idft_shift.dtype == "complex128":
            img_back = np.fft.ifft2(idft_shift)

            return img_back.real

        else:
            img_back = cv2.idft(idft_shift, flags=cv2.DFT_SCALE)
            img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
            return img_back

    # def part1_fft(self):
    # start = time.process_time()

    # dft_shift, magnitude_spectrum = self.fft_cv2(self.raw_img)
    # img_back = self.ifft_cv2(dft_shift)

    # qimg = self.get_qimg(magnitude_spectrum)
    # self.ui.label_img_2.setPixmap(QPixmap.fromImage(qimg))
    # qimg = self.get_qimg(img_back)
    # self.ui.label_img_7.setPixmap(QPixmap.fromImage(qimg))

    # end = time.process_time()
    # self.ui.textEdit.setText(f"Process time: {(end-start):0.10f} s")

    def part1_wavy(
        self,
    ):
        img = self.raw_img
        img_output = np.zeros(img.shape, dtype=img.dtype)

        rows, cols = img.shape
        for i in range(rows):
            for j in range(cols):
                offset_x = int(20.0 * math.sin(2 * 3.14 * i / 150))
                offset_y = int(20.0 * math.cos(2 * 3.14 * j / 150))
                if i + offset_y < rows and j + offset_x < cols:
                    img_output[i, j] = img[(i + offset_y) % rows, (j + offset_x) % cols]
                else:
                    img_output[i, j] = 0

        qimg = self.get_qimg(img_output, 200, 200)
        self.ui.label_img_7.setPixmap(QPixmap.fromImage(qimg))

    def part1_circular(self):
        pass

    def part1_trapezoidal(self):
        pass

    def part3_hough(self):
        img = self.raw_img
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(img, 50, 150, apertureSize=3)
        qimg = self.get_qimg(edges, 200, 200)
        self.ui.label_img_6.setPixmap(QPixmap.fromImage(qimg))

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=100,
            maxLineGap=10,
        )
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        qimg = self.get_qimg(img, 200, 200)
        self.ui.label_img_9.setPixmap(QPixmap.fromImage(qimg))

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=40,
            minLineLength=100,
            maxLineGap=10,
        )  # (72, 1, 4)

        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        _, thresh = cv2.threshold(edges, 240, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )
        area = []
        for contour in contours:
            a = cv2.contourArea(contour)
            area.append(abs(a))

        top_index = area.index(max(area))
        top_area = area[top_index]
        perimeter = cv2.arcLength(contours[top_index], closed=True)
        self.ui.textEdit_2.setText(
            f"Area: {top_area*0.25:} mm^2\n\nPerimeter: {perimeter*0.5} mm"
        )
