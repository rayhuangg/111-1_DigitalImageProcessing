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
        src = cv2.imread("library(854_480).jpg", 0)
        print(src.shape)

        start = time.process_time()
        # 將影像進行float轉換才能進行dft
        result = cv2.dft(np.float32(src), flags=cv2.DFT_COMPLEX_OUTPUT)
        # 將spectrum平移到中心
        dft_shift = np.fft.fftshift(result)
        print(dft_shift.shape)
        s1 = np.log(np.abs(dft_shift))
        center_point = tuple (map (lambda x: (x - 1) / 2, s1.shape))

        mask = np.zeros((dft_shift.shape[0], dft_shift.shape[1], 2), np.uint8)
        # a = np.log(np.abs(dft_shift))
        ci, cj = mask.shape[0]//2, mask.shape[1]//2
        print(ci,cj)

        print(center_point)

    def main(self):
        self.test()

test = test_qt()
test.main()