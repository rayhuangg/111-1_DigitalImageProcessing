
#%%# -*- coding: utf-8 -*-
# ex02

import cv2, copy
import numpy as np

img = cv2.imread('lena.jpg')

# split color channel (opencv: BGR)
B = img[:,:,0]
G = img[:,:,1]
R = img[:,:,2]
# gray_A = np.zeros_like(R) # 複製圖像大小
# gray_B = np.zeros_like(G)

# for i in range(img.shape[0]):
#     for n in range(img.shape[1]):
#         # gray_A[i,n] = (int(R[i,n]) + int(G[i,n]) + int(B[i,n])) // 3.0
#         gray_B[i,n] = int(0.299 * int(R[i,n]) + 0.587 * int(G[i,n]) + 0.114 * int(B[i,n]))

gray_A = np.array((R/3 + G/3.0 + B/3.0), dtype=np.uint8)
gray_B = np.array((0.299*(R) + 0.587*(G) + 0.114*(B)), dtype=np.uint8)
diff = gray_A-gray_B


cv2.imshow('img', img)
cv2.imshow('A', gray_A)
cv2.imshow('B', gray_B)
cv2.imshow('diff', diff)
cv2.waitKey(0)
cv2.destroyAllWindows()


# %%