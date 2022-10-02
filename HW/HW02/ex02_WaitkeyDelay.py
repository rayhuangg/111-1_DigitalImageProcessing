
#%%# -*- coding: utf-8 -*-
# ex02

import cv2
import numpy as np

img = cv2.imread("lena.jpg")

# split color channel
R = img[:,:,0]
G = img[:,:,1]
B = img[:,:,2]
gray_A = np.zeros_like(R) # 複製大小
gray_B = np.zeros_like(R)

for i in range(img.shape[0]):
    for n in range(img.shape[1]):
        gray_A[i,n] = (int(R[i,n]) + int(G[i,n]) + int(B[i,n])) // 3
        gray_B[i,n] = int(0.3 * int(R[i,n]) + 0.1 * int(G[i,n]) + 0.1 * int(B[i,n]))

print(len(img.shape))
print(len(R.shape))


# print(img.shape)
# print(R.shape)
# print(gray_B.shape[0])

# # img = cv2.resize(img, (200,150))
# # img = cv2.resize(img, (200,150))
# # img = cv2.resize(img, (200,150))

# print(img.shape)

# cv2.imshow('A', gray_A)
# cv2.imshow('B', gray_B)
# cv2.imshow('img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# img_name = '123'
# print(type(img_name))
# if type(img_name) == str: # 路徑的話就做imread,否則直接用
#     print("yes")
# %%
