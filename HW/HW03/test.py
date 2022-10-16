
#%%# -*- coding: utf-8 -*-
# ex02

import cv2, copy, math
import numpy as np

img = cv2.imread('lena.jpg')
B, G, R = cv2.split(img)
gray = np.array((R/3 + G/3 + B/3), dtype=np.uint8)


k = np.array([[1,1,1],
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

img_new = convolution(k , gray)
img_new = np.uint8(img_new)

print(gray.shape)
print(img_new.shape)

# cv2.imshow('img', gray)
# cv2.imshow('img_n', img_new)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# %%


class Conv3x3:
    # ...
    def __init__(self, num_filters):
        self.num_filters = num_filters

        # filters is a 3d array with dimensions (num_filters, 3, 3)
        # We divide by 9 to reduce the variance of our initial values
        self.filters = np.random.randn(num_filters, 3, 3) / 9


    def iterate_regions(self, image):
        '''
        Generates all possible 3x3 image regions using valid padding.
        - image is a 2d numpy array
        '''
        h, w = image.shape

        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image[i:(i + 3), j:(j + 3)]
                yield im_region, i, j
        # 将 im_region, i, j 以 tuple 形式存储到迭代器中
        # 以便后面遍历使用

    # zero padding
    def padding(self, data, n):
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
            # 卷积运算，点乘再相加，ouput[i, j] 为向量，8 层
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))
        # 最后将输出数据返回，便于下一层的输入使用
        return output
