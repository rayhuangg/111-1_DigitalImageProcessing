
#%%# -*- coding: utf-8 -*-
# ex02

import cv2, copy, math
import numpy as np

img = cv2.imread('lena.jpg')
B, G, R = cv2.split(img)
gray = np.array((R/3 + G/3 + B/3), dtype=np.uint8)


k = np.array([[1,1,1],
              [1,1,1],
              [1,1,1]] / 9.0)


## Calculate weight by distance
for x in range(1, gray.shape[0]-1):
    for y in range(1, gray.shape[1]-1):
        pass

cv2.imshow('img', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()


# %%