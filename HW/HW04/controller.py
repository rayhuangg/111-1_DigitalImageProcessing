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
        self.ui.button_fft.clicked.connect(self.part1_fft)
        self.ui.button_go.clicked.connect(self.part2_select_filter)

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
            img = cv2.imread(img, 0) # gray
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
        B = img[:,:,0]
        G = img[:,:,1]
        R = img[:,:,2]
        return np.array((R/3 + G/3 + B/3), dtype=np.uint8)


    def fft_cv2(self, img):
        if len(img.shape) == 3:
            img = self.grayscale(img)

        # 將影像進行float轉換才能進行dft
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        # 將spectrum平移到中心
        dft_shift = np.fft.fftshift(dft)
        # 將spectrum複數空間轉換為 0-255 區間
        magnitude_spectrum = 20 * np.log(cv2.magnitude(x=dft_shift[:,:,0], y=dft_shift[:,:,1]))

        return dft_shift, magnitude_spectrum


    # def ifft_cv2(self, dft_shift):
    #     idft_shift = np.fft.ifftshift(dft_shift)
    #     img_back = cv2.idft(idft_shift, flags=cv2.DFT_SCALE )
    #     # img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    #     return img_back


    def ifft_cv2(self, dft_shift):
        # 反向平移
        idft_shift = np.fft.ifftshift(dft_shift)
        if idft_shift.dtype ==  "complex128":
            img_back = np.fft.ifft2(idft_shift)

            return img_back.real

        else:
            img_back = cv2.idft(idft_shift, flags=cv2.DFT_SCALE)
            img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
            return img_back



    def part1_fft(self):
        start = time.process_time()

        dft_shift, magnitude_spectrum = self.fft_cv2(self.raw_img)
        img_back = self.ifft_cv2(dft_shift)

        qimg = self.get_qimg(magnitude_spectrum)
        self.ui.label_img_2.setPixmap(QPixmap.fromImage(qimg))
        qimg = self.get_qimg(img_back)
        self.ui.label_img_7.setPixmap(QPixmap.fromImage(qimg))

        end = time.process_time()
        self.ui.textEdit.setText(f"Process time: {(end-start):0.10f} s")


    # 選擇要用哪個filter
    def part2_select_filter(self):
        if (self.ui.comboBox_filter.currentText()) == "Ideal lowpass filter":
            self.idal_filter(type="low")
        elif (self.ui.comboBox_filter.currentText()) == "Ideal highpass filter":
            self.idal_filter(type="high")
        elif (self.ui.comboBox_filter.currentText()) == "Butterworth lowpass filter":
            self.butterworth_filter(type="low")
        elif (self.ui.comboBox_filter.currentText()) == "Butterworth highpass filter":
            self.butterworth_filter(type="high")
        elif (self.ui.comboBox_filter.currentText()) == "Gaussian lowpass filter":
            self.gaussian_filter(type="low")
        elif (self.ui.comboBox_filter.currentText()) == "Gaussian highpass filter":
            self.gaussian_filter(type="high")


    def idal_filter(self, type="low"):
        # https://blog.csdn.net/qq_38463737/article/details/118682500
        dft_shift, magnitude_spectrum = self.fft_cv2(self.raw_img)
        # 設置截止頻率
        d0 = self.ui.horizontalSlider_cutoff.value()
        rows, cols = self.raw_img.shape[0], self.raw_img.shape[1]
        crow, ccol = int(rows/2), int(cols/2) # mask中心位置

        if type == "low":
            mask = np.zeros((rows, cols, 2), np.uint8)
            mask[crow-d0 : crow+d0, ccol-d0 : ccol+d0] = 1 # 設定mask
            # mask和頻譜圖像相乘濾波
            dft_shift = dft_shift * mask
        elif type == "high":
            # mask = np.ones((rows, cols, 2), np.uint8)
            # mask[crow-cut_off : crow+cut_off, ccol-cut_off : ccol+cut_off] = 0 # 設定mask
            dft_shift[crow-d0 : crow+d0, ccol-d0 : ccol+d0] = 0

        ## FIXME 為何要加上abs/clip才不會出現奇怪黑白色塊還未知
        # img_back = np.abs(self.ifft_cv2(dft_shift))
        img_back = np.clip(self.ifft_cv2(dft_shift), 0, 255)
        qimg = self.get_qimg(img_back)
        self.ui.label_img_4.setPixmap(QPixmap.fromImage(qimg))


    def butterworth_filter(self, type="low"):
        dft_shift, spectrum = self.fft_cv2(self.raw_img)

        ##### FIXME mask type用uint8就不行..........
        # mask = np.zeros((dft_shift.shape[0], dft_shift.shape[1], 2), dtype=np.uint8)
        # print(mask[0][0])
        # print(mask.dtype)
        mask = np.zeros((dft_shift.shape[0], dft_shift.shape[1], 2))
        # print(mask[0][0])
        # print(mask.dtype)

        n = 2
        d0 = self.ui.horizontalSlider_cutoff.value()
        ci, cj = mask.shape[0]//2, mask.shape[1]//2

        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if type == "low":
                    denominator = 1 + ((math.sqrt((i - ci)**2 + (j - cj)**2) / d0) ** (2*n))
                elif type == "high":
                    denominator = 1 + (d0 / ((math.sqrt((i - ci)**2 + (j - cj)**2)) + 0.00000000001) ** (2*n)) # 避免分母為0

                mask[i, j] = 1 / denominator

        dft_shift = dft_shift * mask
        # img_back = np.abs(self.ifft_cv2(dft_shift))
        img_back = np.clip(self.ifft_cv2(dft_shift), 0, 255)
        qimg = self.get_qimg(img_back)
        self.ui.label_img_4.setPixmap(QPixmap.fromImage(qimg))


    def gaussian_filter(self, type="low"):
        dft_shift, magnitude_spectrum = self.fft_cv2(self.raw_img)
        mask = np.zeros((dft_shift.shape[0], dft_shift.shape[1], 2))
        d0 = self.ui.horizontalSlider_cutoff.value()
        ci, cj = mask.shape[0]//2, mask.shape[1]//2

        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if type == "low":
                    power = -1 * ((math.sqrt((i - ci)**2 + (j - cj)**2) ** 2)) / (2 * (d0**2))
                    mask[i, j] = math.exp(power)
                elif type == "high":
                    power = -1 * ((math.sqrt((i - ci)**2 + (j - cj)**2) ** 2)) / (2 * (d0**2))
                    mask[i, j] = 1 - (math.exp(power))

        dft_shift = dft_shift * mask
        img_back = np.clip(self.ifft_cv2(dft_shift), 0, 255)
        qimg = self.get_qimg(img_back)
        self.ui.label_img_4.setPixmap(QPixmap.fromImage(qimg))


    # https://zhuanlan.zhihu.com/p/515812634
    def homomorphic_filter(self):
        dft_shift, magnitude_spectrum = self.fft_cv2(self.raw_img)
        mask = np.zeros((dft_shift.shape[0], dft_shift.shape[1], 2))
        ci, cj = mask.shape[0]//2, mask.shape[1]//2
        gh = self.ui.horizontalSlider_gh.value()
        gl = self.ui.horizontalSlider_gl.value()
        d0 = self.ui.horizontalSlider_d0.value()
        c = self.ui.horizontalSlider_c.value()

        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                d = math.sqrt((i - ci)**2 + (j - cj)**2)
                power = (-c * (d ** 2 / d0))
                mask[i, j] = (gh - gl) * (1 - math.exp(power)) + gl

        dft_shift = dft_shift * mask
        # img_back = np.abs(self.ifft_cv2(dft_shift))
        img_back = np.clip(self.ifft_cv2(dft_shift), 0, 255)
        qimg = self.get_qimg(img_back)
        self.ui.label_img_6.setPixmap(QPixmap.fromImage(qimg))


    def part4(self, noise=False):
        self.ui.textEdit__blur_type.setText("Type: "+str(self.ui.comboBox_blur.currentText()))
        a, b, T = 0.1, 0.1, 1
        dft_shift, magnitude_spectrum = self.fft_cv2(self.raw_img)
        mask = np.zeros((dft_shift.shape[0], dft_shift.shape[1], 2), dtype=complex) # 設定複數形式

        # blur
        for u in range(mask.shape[0]):
            for v in range(mask.shape[1]):
                denominator = math.pi * (u*a + v*b)
                power = -1j * math.pi * (u*a + v*b)
                mask[u,v] = (T / (denominator+ 0.00000000001)) * math.sin(math.pi * (u*a + v*b)) * cmath.exp(power) # 避免分母0

        dft_shift = dft_shift * mask
        # img_back = (self.ifft_cv2(dft_shift))
        img_back = np.abs(self.ifft_cv2(dft_shift))
        # img_back = self.ifft_cv2(dft_shift)
        qimg = self.get_qimg(img_back)
        self.ui.label_img_9.setPixmap(QPixmap.fromImage(qimg))