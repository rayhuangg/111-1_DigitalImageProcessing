#%%
import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

#%%

# img = cv2.imread('Part 3 Image/rects.bmp')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# cv2.imshow('edges', edges)
# lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=10) # (72, 1, 4)

# for line in lines:
#     x1, y1, x2, y2 = line[0]
#     cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # green

# cv2.imshow('Hough Lines', img)
# k = cv2.waitKey(0)
# cv2.destroyAllWindows()


#%%
# https://subscription.packtpub.com/book/data/9781788396905/1/ch01lvl1sec19/image-warping
img = cv2.imread("Part 1 Image/IP_dog.bmp", cv2.IMREAD_GRAYSCALE)
img_output = np.zeros(img.shape, dtype=img.dtype)

rows, cols = img.shape
for i in range(rows):
    for j in range(cols):
        offset_x = int(20.0 * math.sin(2 * 3.14 * i / 150))
        offset_y = int(20.0 * math.cos(2 * 3.14 * j / 150))
        if i+offset_y < rows and j+offset_x < cols:
            img_output[i,j] = img[(i+offset_y)%rows,(j+offset_x)%cols]
        else:
            img_output[i,j] = 0


# cv2.imshow('Multidirectional wave', img_output)
plt.imshow(img_output, cmap="gray")
plt.show()
