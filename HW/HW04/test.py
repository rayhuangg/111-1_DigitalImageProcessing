from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog

import cv2, os, copy, math, time
import numpy as np
from matplotlib import pyplot as plt

class test_qt:
    def fft(self, img=0):
        src = cv2.imread("library(854_480).jpg", 0)

        start = time.process_time()
        # 將影像進行float轉換才能進行dft
        result = cv2.dft(np.float32(src), flags=cv2.DFT_COMPLEX_OUTPUT)
        # 將spectrum平移到中心
        dft_shift = np.fft.fftshift(result)
        # 將spectrum複數轉換為 0-255 區間
        result1 = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))


        plt.subplot(121), plt.imshow(src, 'gray'), plt.title('oreginal')
        plt.axis('off')
        plt.subplot(122), plt.imshow(result1, 'gray'), plt.title('FFT')
        plt.axis('off')
        plt.show()
        end = time.process_time()
        print(f"Process time: {(end-start):0.10f} s")
        return result1

    def test(self):
        img = cv2.imread('C1HW04_IMG01_2022.jpg',0)
        # f = np.fft.fft2(img)
        f = cv2.dft(np.float32(img), flags=cv2.DFT_REAL_OUTPUT)

        fshift = np.fft.fftshift(f)
        f_ishift = np.fft.ifftshift(fshift)
        print(f_ishift.shape)
        d_shift = np.array(np.dstack([f_ishift.real,f_ishift.imag]))
        print(d_shift.shape)
        img_back = cv2.idft(d_shift)
        img = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

        plt.imshow(img)
        plt.show()
    def main(self):
        self.test()

test = test_qt()
test.main()