from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog
from PyQt5 import QtCore, QtGui, QtWidgets

import cv2, os, copy, math, cmath, time
import numpy as np
from matplotlib import pyplot as plt


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


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(801, 606)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.tabWidget.setFont(font)
        self.tabWidget.setElideMode(QtCore.Qt.ElideLeft)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.button_original = QtWidgets.QPushButton(self.tab)
        self.button_original.setGeometry(QtCore.QRect(10, 110, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.button_original.setFont(font)
        self.button_original.setObjectName("button_original")
        self.label_img_1 = QtWidgets.QLabel(self.tab)
        self.label_img_1.setGeometry(QtCore.QRect(160, 30, 200, 200))
        self.label_img_1.setAlignment(QtCore.Qt.AlignCenter)
        self.label_img_1.setObjectName("label_img_1")
        self.button_trapezoidal = QtWidgets.QPushButton(self.tab)
        self.button_trapezoidal.setGeometry(QtCore.QRect(390, 100, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.button_trapezoidal.setFont(font)
        self.button_trapezoidal.setObjectName("button_trapezoidal")
        self.button_wavy = QtWidgets.QPushButton(self.tab)
        self.button_wavy.setGeometry(QtCore.QRect(10, 400, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.button_wavy.setFont(font)
        self.button_wavy.setObjectName("button_wavy")
        self.button_circular = QtWidgets.QPushButton(self.tab)
        self.button_circular.setGeometry(QtCore.QRect(400, 410, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.button_circular.setFont(font)
        self.button_circular.setObjectName("button_circular")
        self.button_file_1 = QtWidgets.QPushButton(self.tab)
        self.button_file_1.setGeometry(QtCore.QRect(20, 20, 81, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.button_file_1.setFont(font)
        self.button_file_1.setObjectName("button_file_1")
        self.label_img_2 = QtWidgets.QLabel(self.tab)
        self.label_img_2.setGeometry(QtCore.QRect(550, 30, 200, 200))
        self.label_img_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_img_2.setObjectName("label_img_2")
        self.label_img_7 = QtWidgets.QLabel(self.tab)
        self.label_img_7.setGeometry(QtCore.QRect(160, 260, 200, 200))
        self.label_img_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_img_7.setObjectName("label_img_7")
        self.label_img_8 = QtWidgets.QLabel(self.tab)
        self.label_img_8.setGeometry(QtCore.QRect(550, 270, 200, 200))
        self.label_img_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_img_8.setObjectName("label_img_8")
        self.label_threshold_4 = QtWidgets.QLabel(self.tab)
        self.label_threshold_4.setGeometry(QtCore.QRect(410, 130, 21, 51))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_threshold_4.setFont(font)
        self.label_threshold_4.setTextFormat(QtCore.Qt.MarkdownText)
        self.label_threshold_4.setObjectName("label_threshold_4")
        self.label_threshold_5 = QtWidgets.QLabel(self.tab)
        self.label_threshold_5.setGeometry(QtCore.QRect(410, 440, 21, 51))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_threshold_5.setFont(font)
        self.label_threshold_5.setTextFormat(QtCore.Qt.MarkdownText)
        self.label_threshold_5.setObjectName("label_threshold_5")
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.button_file_2 = QtWidgets.QPushButton(self.tab_2)
        self.button_file_2.setGeometry(QtCore.QRect(30, 50, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.button_file_2.setFont(font)
        self.button_file_2.setObjectName("button_file_2")
        self.label_threshold_9 = QtWidgets.QLabel(self.tab_2)
        self.label_threshold_9.setGeometry(QtCore.QRect(60, 150, 681, 161))
        font = QtGui.QFont()
        font.setFamily("Kristen ITC")
        font.setPointSize(28)
        self.label_threshold_9.setFont(font)
        self.label_threshold_9.setTextFormat(QtCore.Qt.MarkdownText)
        self.label_threshold_9.setObjectName("label_threshold_9")
        self.tabWidget.addTab(self.tab_2, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.button_file_3 = QtWidgets.QPushButton(self.tab_3)
        self.button_file_3.setGeometry(QtCore.QRect(30, 20, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.button_file_3.setFont(font)
        self.button_file_3.setObjectName("button_file_3")
        self.label_brightness = QtWidgets.QLabel(self.tab_3)
        self.label_brightness.setGeometry(QtCore.QRect(220, 140, 81, 16))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_brightness.setFont(font)
        self.label_brightness.setText("")
        self.label_brightness.setObjectName("label_brightness")
        self.label_threshold_7 = QtWidgets.QLabel(self.tab_3)
        self.label_threshold_7.setGeometry(QtCore.QRect(40, 110, 101, 51))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_threshold_7.setFont(font)
        self.label_threshold_7.setTextFormat(QtCore.Qt.MarkdownText)
        self.label_threshold_7.setObjectName("label_threshold_7")
        self.label_img_5 = QtWidgets.QLabel(self.tab_3)
        self.label_img_5.setGeometry(QtCore.QRect(30, 180, 200, 200))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_img_5.setFont(font)
        self.label_img_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_img_5.setObjectName("label_img_5")
        self.label_img_6 = QtWidgets.QLabel(self.tab_3)
        self.label_img_6.setGeometry(QtCore.QRect(300, 60, 200, 200))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_img_6.setFont(font)
        self.label_img_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_img_6.setObjectName("label_img_6")
        self.label_threshold_8 = QtWidgets.QLabel(self.tab_3)
        self.label_threshold_8.setGeometry(QtCore.QRect(300, 0, 161, 51))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_threshold_8.setFont(font)
        self.label_threshold_8.setTextFormat(QtCore.Qt.MarkdownText)
        self.label_threshold_8.setObjectName("label_threshold_8")
        self.textEdit_2 = QtWidgets.QTextEdit(self.tab_3)
        self.textEdit_2.setGeometry(QtCore.QRect(520, 310, 201, 101))
        self.textEdit_2.setTabletTracking(False)
        self.textEdit_2.setObjectName("textEdit_2")
        self.label_img_9 = QtWidgets.QLabel(self.tab_3)
        self.label_img_9.setGeometry(QtCore.QRect(290, 310, 200, 200))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_img_9.setFont(font)
        self.label_img_9.setAlignment(QtCore.Qt.AlignCenter)
        self.label_img_9.setObjectName("label_img_9")
        self.label_threshold_12 = QtWidgets.QLabel(self.tab_3)
        self.label_threshold_12.setGeometry(QtCore.QRect(290, 260, 161, 51))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.label_threshold_12.setFont(font)
        self.label_threshold_12.setTextFormat(QtCore.Qt.MarkdownText)
        self.label_threshold_12.setObjectName("label_threshold_12")
        self.tabWidget.addTab(self.tab_3, "")
        self.gridLayout.addWidget(self.tabWidget, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 801, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.button_original.setText(_translate("MainWindow", "Original"))
        self.label_img_1.setText(_translate("MainWindow", "img"))
        self.button_trapezoidal.setText(_translate("MainWindow", "Trapezoidal"))
        self.button_wavy.setText(_translate("MainWindow", "Wavy"))
        self.button_circular.setText(_translate("MainWindow", "Circular"))
        self.button_file_1.setText(_translate("MainWindow", "choose img"))
        self.label_img_2.setText(_translate("MainWindow", "img"))
        self.label_img_7.setText(_translate("MainWindow", "img"))
        self.label_img_8.setText(_translate("MainWindow", "img"))
        self.label_threshold_4.setText(_translate("MainWindow", "X"))
        self.label_threshold_5.setText(_translate("MainWindow", "X"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Part1"))
        self.button_file_2.setText(_translate("MainWindow", "choose img"))
        self.label_threshold_9.setText(_translate("MainWindow", "I haven\'t finished this part yet. Sorry."))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Part2"))
        self.button_file_3.setText(_translate("MainWindow", "choose img"))
        self.label_threshold_7.setText(_translate("MainWindow", "Original"))
        self.label_img_5.setText(_translate("MainWindow", "img"))
        self.label_img_6.setText(_translate("MainWindow", "img"))
        self.label_threshold_8.setText(_translate("MainWindow", "Edge"))
        self.label_img_9.setText(_translate("MainWindow", "img"))
        self.label_threshold_12.setText(_translate("MainWindow", "Hough transform"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "Part3"))




if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow_controller()
    window.show()
    sys.exit(app.exec_())