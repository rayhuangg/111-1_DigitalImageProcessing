
#%%# -*- coding: utf-8 -*-
# ex02

import cv2, copy, math
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('lena.jpg')
B, G, R = cv2.split(img)
gray = np.array((R/3 + G/3 + B/3), dtype=np.uint8)


kernal = np.array([[1,1,1],
                   [1,1,1],
                   [1,1,1]]) / 9


def convolution(k, data):
    h, w = data.shape
    img_new = []
    output = np.zeros((h - 2, w - 2,))

    for i in range(h-3):
        line = []
        for j in range(w-3):
            a = data[i:i+3,j:j+3]
            # # line.append(np.sum(np.mul gion * self.filters, axis=(1, 2))
        img_new.append(line)
    return np.array(img_new)

# img_new = convolution(kernal , gray)
# img_new = np.clip(np.uint8(img_new), 0, 255)

# print(gray.shape)
# print(img_new.shape)

# cv2.imshow('img', gray)
# cv2.imshow('img_n', img_new)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# %%


class Conv3x3:
    # ...
    def __init__(self, num_filters=3):
        self.num_filters = num_filters

        # filters is a 3d array with dimensions (num_filters, 3, 3)
        # We divide by 9 to reduce the variance of our initial values
        # self.filters = np.random.randn(num_filters, 3, 3) / 9
        self.filters =  np.array([[1,1,1],
                   [1,1,1],
                   [1,1,1]]) / 9

    # get resize image
    def resize(self, img, width=200, height=150):
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

    def forward(self, input):
        '''
        Performs a forward pass of the conv layer using the given input.
        Returns a 3d numpy array with dimensions (h, w, num_filters).
        - input is a 2d numpy array
        '''
        # input 为 image，即输入数据
        # output 为输出框架，默认都为 0，都为 1 也可以，反正后面会覆盖
        # input: 28x28
        # output: 26x26x8
        h, w = input.shape
        output = np.zeros((h - 2, w - 2, self.num_filters))

        for im_region, i, j in self.iterate_regions(input):
            # 捲機運算，点乘再相加，ouput[i, j] 为向量，8 层
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))
            output = np.clip(np.uint8(output), 0, 255)
        print(output.shape)
        return output

    def main(self):
        img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
        img = self.resize(img)

        img_c = self.forward(img)


        plt.figure()
        plt.imshow(img_c, cmap='gray')
        plt.show()

        # https://github.com/vzhou842/cnn-from-scratch/blob/master/conv.py

c = Conv3x3()
c.main()