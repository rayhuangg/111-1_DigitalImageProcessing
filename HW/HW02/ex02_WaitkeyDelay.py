
#%%# -*- coding: utf-8 -*-
# ex02

import cv2, copy
import numpy as np

img = cv2.imread("lena.jpg")

brightness_img_add = img * 2.0  + 0
print(type(brightness_img_add))
brightness_img_add = np.clip(brightness_img_add, 0, 255)
brightness_img_add = np.uint8(brightness_img_add)


contrast = 127
brightness = 0
output = img * (contrast/127 + 1) - contrast + brightness
print(type(output))
output = np.clip(output, 0, 255)
output = np.uint8(output)


cv2.imshow('img', img)
cv2.imshow('out', output)
cv2.imshow('add', brightness_img_add)
cv2.waitKey(0)
cv2.destroyAllWindows()


# %%