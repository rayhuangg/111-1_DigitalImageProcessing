
#%%# -*- coding: utf-8 -*-
# ex02

import cv2

img = cv2.imread("lena.jpg")

R = img[:,:,0]
G = img[:,:,1]
B = img[:,:,2]

print(img.shape)
print(R.shape)



# cv2.imshow('R', R)
# cv2.imshow('G', G)
# cv2.imshow('B', B)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# img_name = '123'
# print(type(img_name))
# if type(img_name) == str: # 路徑的話就做imread,否則直接用
#     print("yes")