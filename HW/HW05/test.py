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
        # self.raw_img = cv2.imread("lena.jpg")


    def get_split_channel(self, img):
        return img[:,:,0], img[:,:,2], img[:,:,2]


    def showrgb(self):
        img = self.raw_img
        return img


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

        return result
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

        return hsi


    def rgb_yuv(self, rgb_img ):
        W = np.array([
            [0.299, 0.587, 0.114],
            [-0.148, -0.289, 0.437],
            [0.615, -0.515, -0.100]
        ])
        rgb_Img = rgb_img.copy()
        rgb_Img = rgb_Img.astype(np.float)
        h, w, c = rgb_Img.shape
        for i in range(h):
            for j in range(w):
                rgb_Img[i, j] = np.dot(W, rgb_Img[i, j])
        imc = rgb_Img.astype(np.uint8)
        return imc


    def rgb2lab (self) :
        num = 0
        RGB = [0, 0, 0]
        img = self.raw_img

        for value in img :
            value = float(value) / 255

            if value > 0.04045 :
                value = ( ( value + 0.055 ) / 1.055 ) ** 2.4
            else :
                value = value / 12.92

            RGB[num] = value * 100
            num = num + 1

        XYZ = [0, 0, 0,]

        X = RGB [0] * 0.4124 + RGB [1] * 0.3576 + RGB [2] * 0.1805
        Y = RGB [0] * 0.2126 + RGB [1] * 0.7152 + RGB [2] * 0.0722
        Z = RGB [0] * 0.0193 + RGB [1] * 0.1192 + RGB [2] * 0.9505
        XYZ[ 0 ] = round( X, 4 )
        XYZ[ 1 ] = round( Y, 4 )
        XYZ[ 2 ] = round( Z, 4 )

        XYZ[ 0 ] = float( XYZ[ 0 ] ) / 95.047         # ref_X =  95.047   Observer= 2°, Illuminant= D65
        XYZ[ 1 ] = float( XYZ[ 1 ] ) / 100.0          # ref_Y = 100.000
        XYZ[ 2 ] = float( XYZ[ 2 ] ) / 108.883        # ref_Z = 108.883

        num = 0
        for value in XYZ :

            if value > 0.008856 :
                value = value ** ( 0.3333333333333333 )
            else :
                value = ( 7.787 * value ) + ( 16 / 116 )

            XYZ[num] = value
            num = num + 1

        Lab = [0, 0, 0]

        L = ( 116 * XYZ[ 1 ] ) - 16
        a = 500 * ( XYZ[ 0 ] - XYZ[ 1 ] )
        b = 200 * ( XYZ[ 1 ] - XYZ[ 2 ] )

        Lab [ 0 ] = round( L, 4 )
        Lab [ 1 ] = round( a, 4 )
        Lab [ 2 ] = round( b, 4 )

        return Lab


    def grayscale(self, img):
        B = img[:,:,0]
        G = img[:,:,1]
        R = img[:,:,2]
        return np.array((R/3 + G/3 + B/3), dtype=np.uint8)


    def kmeans(self):
        img = cv2.imread("lena.jpg")
        img = self.grayscale(img)
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(img,'gray')
        plt.title('Original')
        plt.xticks([])
        plt.yticks([])

        #change img(2D) to 1D
        img1 = img.reshape((img.shape[0]*img.shape[1],1))
        img1 = np.float32(img1)

        #define criteria = (type,max_iter,epsilon)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1.0)

        #set flags: hou to choose the initial center
        #---cv2.KMEANS_PP_CENTERS ; cv2.KMEANS_RANDOM_CENTERS
        flags = cv2.KMEANS_RANDOM_CENTERS
        # apply kmenas
        compactness,labels,centers = cv2.kmeans(img1,4,None,criteria,10,flags)

        img2 = labels.reshape((img.shape[0],img.shape[1]))
        plt.subplot(1,2,2)
        plt.imshow(img2,'gray')
        plt.title('Kmeans')
        plt.xticks([])
        plt.yticks([])
        plt.show()




    # im_channel取值範圍：[0,1]
    def f(self, im_channel):
        return np.power(im_channel, 1 / 3) if im_channel > 0.008856 else 7.787 * im_channel + 0.137931


    def anti_f(self, im_channel):
        return np.power(im_channel, 3) if im_channel > 0.206893 else (im_channel - 0.137931) / 7.787
    # endregion


    # region RGB 轉 Lab
    # 畫素值RGB轉XYZ空間，pixel格式:(B,G,R)
    # 返回XYZ空間下的值
    def __rgb2xyz__(self, pixel):
        b, g, r = pixel[0], pixel[1], pixel[2]
        rgb = np.array([r, g, b])
        M = np.array([[0.412453, 0.357580, 0.180423],
                [0.212671, 0.715160, 0.072169],
                [0.019334, 0.119193, 0.950227]])
        # rgb = rgb / 255.0
        # RGB = np.array([gamma(c) for c in rgb])
        XYZ = np.dot(M, rgb.T)
        XYZ = XYZ / 255.0
        return (XYZ[0] / 0.95047, XYZ[1] / 1.0, XYZ[2] / 1.08883)


    def __rgb2xyz2__(self, pixel):
        b, g, r = pixel[0], pixel[1], pixel[2]
        rgb = np.array([r, g, b])
        M = np.array([[0.412453, 0.357580, 0.180423],
                [0.212671, 0.715160, 0.072169],
                [0.019334, 0.119193, 0.950227]])
        # rgb = rgb / 255.0
        # RGB = np.array([gamma(c) for c in rgb])
        XYZ = np.dot(M, rgb.T)
        XYZ = XYZ / 255.0
        return XYZ[0] / 0.95047, XYZ[1] / 1.0, XYZ[2] / 1.08883


    def __xyz2lab__(self, xyz):
        F_XYZ = [self.f(x) for x in xyz]
        L = 116 * F_XYZ[1] - 16 if xyz[1] > 0.008856 else 903.3 * xyz[1]
        a = 500 * (F_XYZ[0] - F_XYZ[1])
        b = 200 * (F_XYZ[1] - F_XYZ[2])
        return (L, a, b)


    def rgb2lab(self, pixel):
        xyz = self.__rgb2xyz__(pixel)
        Lab = self.__xyz2lab__(xyz)
        return Lab


    def rgb2xyz(self, pixel):
        xyz = self.__rgb2xyz2__(pixel)
        return xyz


    # endregion

    # region Lab 轉 RGB
    def __lab2xyz__(self, Lab):
        fY = (Lab[0] + 16.0) / 116.0
        fX = Lab[1] / 500.0 + fY
        fZ = fY - Lab[2] / 200.0

        x = self.anti_f(fX)
        y = self.anti_f(fY)
        z = self.anti_f(fZ)

        x = x * 0.95047
        y = y * 1.0
        z = z * 1.0883

        return (x, y, z)


    def __xyz2rgb(self, xyz):
        xyz = np.array(xyz)
        xyz = xyz * 255
        M = np.array([[0.412453, 0.357580, 0.180423],
                [0.212671, 0.715160, 0.072169],
                [0.019334, 0.119193, 0.950227]])
        rgb = np.dot(np.linalg.inv(M), xyz.T)
        # rgb = rgb * 255
        rgb = np.uint8(np.clip(rgb, 0, 255))
        return rgb


    def Lab2RGB(self, Lab):
        xyz = self.__lab2xyz__(Lab)
        rgb = self.__xyz2rgb(xyz)
        return rgb


    def part1(self):
        rgb = self.showrgb()
        cmy = self.rgb2cmy()
        hsi = self.rgb2hsi()
        yuv = self.rgb_yuv(self.raw_img)

        img = self.raw_img
        w = img.shape[0]
        h = img.shape[1]

        xyz = np.zeros((w,h,3))
        lab = np.zeros((w,h,3))

        # get lab image
        for i in range(w):
            for j in range(h):
                Lab = self.rgb2lab(img[i,j])
                lab[i, j] = (Lab[0], Lab[1], Lab[2])

        # get xyz image
        for i in range(w):
            for j in range(h):
                xyz[i, j] = self.rgb2xyz(img[i,j])
                # xyz[i, j] = img[0], img[1], img[2]


        plt.figure()
        plt.subplot(3,2,1)
        plt.imshow(rgb), plt.title('RGB'), plt.xticks([]), plt.yticks([])

        plt.subplot(3,2,2)
        plt.imshow(cmy), plt.title('CMY'), plt.xticks([]), plt.yticks([])

        plt.subplot(3,2,3)
        plt.imshow(hsi), plt.title('HSI'), plt.xticks([]), plt.yticks([])

        plt.subplot(3,2,4)
        plt.imshow(xyz), plt.title('xyz'), plt.xticks([]), plt.yticks([])

        plt.subplot(3,2,5)
        plt.imshow(lab), plt.title('lab'), plt.xticks([]), plt.yticks([])

        plt.subplot(3,2,6)
        plt.imshow(yuv), plt.title('yuv'), plt.xticks([]), plt.yticks([])

        plt.show()


    def main(self):
        self.part1()
        self.kmeans()


test = test_qt()
test.main()