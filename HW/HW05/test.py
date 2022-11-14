# https://cloud.tencent.com/developer/article/1784017
# https://github.com/SVLaursen/Python-RGB-to-HSI/blob/514a02485aae72e84c08a2faa6be6d1d3ed9c65f/converter.py#L5

# from PyQt5 import QtCore, QtWidgets
# from PyQt5.QtGui import QImage, QPixmap
# from PyQt5.QtWidgets import QFileDialog


import cv2, os, copy, math, time
import numpy as np
from matplotlib import pyplot as plt

class test_qt:
    def __init__(self) -> None:
        self.raw_img = cv2.imread("HW05-Part 3-02.bmp")
        self.raw_img = cv2.imread("lena.jpg")


    def get_split_channel(self, img):
        return img[:,:,0], img[:,:,2], img[:,:,2]


    def showrgb(self):
        img = self.raw_img


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

        # plt.imshow(result)
        # plt.show()


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

        plt.imshow(hsi)
        plt.show()


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



    def main(self):
        self.rgb2xyz()

test = test_qt()
test.main()