#%%
import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

#%%

# img = cv2.imread('Part 3 Image/rects.bmp')
img = cv2.imread('card.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)


lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=40, minLineLength=100, maxLineGap=10) # (72, 1, 4)

for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

_, thresh = cv2.threshold(edges, 240, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
area = []
topk_contours =[]
for contour in contours:
    a = cv2.contourArea(contour)
    area.append(abs(a))

top_index = area.index(max(area))
area_sort = sorted(area)

# num = 5 #取最大面積的個數
# for i in range(num):
#     top = area.index(max(area))
#     area.pop(top)
#     topk_contours.append(contours[top])

top_index = area.index(max(area))
area_sort = sorted(area)
second_index = area.index(area_sort[1]) # find second large area

mask_img = cv2.drawContours(gray, contours, -1, (255, 255, 255), 1)

top_area = area[second_index]
perimeter = cv2.arcLength(contours[second_index], closed=True)
print("area:",top_area*0.25, "mm^2")
print("perimeter:", perimeter*0.5, "mm")

plt.imshow(mask_img, cmap="gray")
plt.show()





#%%
# https://subscription.packtpub.com/book/data/9781788396905/1/ch01lvl1sec19/image-warping
# img = cv2.imread("Part 1 Image/IP_dog.bmp", cv2.IMREAD_GRAYSCALE)
# img_output = np.zeros(img.shape, dtype=img.dtype)

# rows, cols = img.shape
# for i in range(rows):
#     for j in range(cols):
#         offset_x = int(20.0 * math.sin(2 * 3.14 * i / 150))
#         offset_y = int(20.0 * math.cos(2 * 3.14 * j / 150))
#         if i+offset_y < rows and j+offset_x < cols:
#             img_output[i,j] = img[(i+offset_y)%rows,(j+offset_x)%cols]
#         else:
#             img_output[i,j] = 0


# # cv2.imshow('Multidirectional wave', img_output)
# plt.imshow(img_output, cmap="gray")
# plt.show()
