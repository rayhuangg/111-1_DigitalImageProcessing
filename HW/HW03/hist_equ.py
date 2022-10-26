#%%
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('lena.jpg', 0)

def get_histogram(img):
    hist = np.zeros(256)
    # 計算各數值數量並儲存
    for i in np.unique(img):
        hist[i] = np.bincount(img.flatten())[i]

    return hist


# calculate histogram
hists = get_histogram(img)
plt.subplot(2,2,1)
plt.imshow(img, cmap='gray')

plt.subplot(2,2,2)
x_axis = np.arange(256)
plt.bar(x_axis, hists)


# caculate cdf
hists_cumsum = np.cumsum(hists)
const = 255 / img.size
hists_cdf = np.uint8(const * hists_cumsum)

# mapping
img_eq = hists_cdf[img]


plt.subplot(2,2,3)
plt.imshow(img_eq, cmap='gray')

plt.subplot(2,2,4)
plt.bar(x_axis, get_histogram(img_eq))
plt.show()