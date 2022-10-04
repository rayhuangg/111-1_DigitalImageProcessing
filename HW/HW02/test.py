
#%%# -*- coding: utf-8 -*-
# ex02


import cv2, copy, math
import numpy as np

img = cv2.imread('lena.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


gray = copy.deepcopy(gray)

enlarge = np.zeros_like(gray)
# enlarge = np.array([[]])

## Calculate weight by distance
for x in range(1, gray.shape[0]-1):
    for y in range(1, gray.shape[1]-1):
        a = x - math.floor(x)
        b = y - math.floor(y)

        enlarge[x,y] = a*b*gray[x+1,y+1] \
                        + (1-a)*b*gray[x,y+1] \
                        + a*(1-b)*gray[x+1,y] \
                        + (1-a)*(1-b)*gray[x,y]





cv2.imshow('img', gray)
cv2.imshow('enlarge', enlarge)
# cv2.imshow('B', gray_B)
# cv2.imshow('diff', diff)
cv2.waitKey(0)
cv2.destroyAllWindows()



# %%