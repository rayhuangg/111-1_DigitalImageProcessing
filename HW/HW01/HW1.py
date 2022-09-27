#%%
import copy
import numpy as np
from matplotlib import pyplot as plt

# import matplotlib
# matplotlib.use('TKAgg')

import os
import qt5_applications
dirname = os.path.dirname(qt5_applications.__file__)
plugin_path = os.path.join(dirname, 'Qt', 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path

# 繪製圖像直方圖
def plot_histogram(img) :
    hist = np.zeros(32)

    # 計算各數值數量並儲存
    for i in np.unique(img):
        hist[i] = np.bincount(img.flatten())[i]

    # 利用長條圖繪出顯示
    x_axis = np.arange(32)
    plt.bar(x_axis, hist)
    # plt.show()


# 繪製圖片
def plot_64(img):
    plt.imshow(img, cmap='gray')
    # plt.show()``


# 讀取原始資料並回傳ndarray
def read_raw_data(filename='LINCOLN.64'):

    with open(filename, 'r') as data:
        data_64 = data.readlines()

    np_64 = np.zeros(shape=(64,64)) # 建立64*64 np array
    np_64 = np.uint8(np_64) # 轉為8bit才可以正常顯示

    # 將64文字資料轉為0~31數值
    for row, line in enumerate(data_64):
        for col, character in enumerate(line):
            if str(character).isdigit():
                character = ord(character) - 48 # ascii -48 即可以將數字對應回原始8bit，ord(0)=48

            # 換行、結束符號即跳過
            elif character == '\n' or character == '\x1a':
                continue

            # 非數字即為大寫英文
            else:
                character = ord(character) - 55 # ascii -55 即可以將字母對應回原始8bit，ord(A)=65

            np_64[row][col] = character
    return np_64


# 使用matplotlib 一次將照片與直方圖顯示
def plot_2_subplot(img, title=""):
    plt.figure(figsize=(6,4))

    # plot 1
    plt.subplot(1, 2, 1)
    plot_64(img=img)
    plt.title("img")

    # plot 2:
    plt.subplot(1, 2, 2)
    plot_histogram(img=img)
    plt.title("histgram")

    plt.suptitle(title)
    plt.subplots_adjust(wspace=0.4)
    plt.show()



#===========  part 1 =========+============

# TODO 記得取消註解

plot_2_subplot(read_raw_data("LISA.64"), title="File name: LISA.64")
plot_2_subplot(read_raw_data("JET.64"), title="File name: JET.64")
plot_2_subplot(read_raw_data("LIBERTY.64"), title="File name: LIBERTY.64")
plot_2_subplot(read_raw_data("LINCOLN.64"), title="File name: LINCOLN.64")


#===========  part 2-1 =====================

np_64 = read_raw_data(filename='LINCOLN.64')
new_64 = copy.deepcopy(np_64) # deep copy才能維持原始資料
offset = -15 # 照片各像素增減數值

# 數值加減計算，並設定上下限
for row, n in enumerate(new_64):
    for col, _ in enumerate(n):
        if (new_64[row][col] + offset) >= 31:  # 設定上限
            new_64[row][col] = 31

        elif (new_64[row][col] + offset) <= 0: # 設定下限
            new_64[row][col] = 0

        else:
            new_64[row][col] += offset

plot_2_subplot(img=new_64, title="Task: Subtract 15 on every pixel")


#===========  part 2-2 =====================

np_64 = read_raw_data(filename='LINCOLN.64')
new_64 = copy.deepcopy(np_64) # deep copy才能維持原始資料
time_constant = 1.85 # 照片各像素乘上的倍數

# 數值乘法計算，並設定上下限
for row, n in enumerate(new_64):
    for col, _ in enumerate(n):
        if (new_64[row][col] * time_constant) >= 31: # 設定上限
            new_64[row][col] = 31
        else:
            new_64[row][col] *= time_constant

plot_2_subplot(img=new_64, title="Task: Multiply 1.75 on every pixel")


#===========  part 2-3 =====================

np_64_1 = read_raw_data(filename='LINCOLN.64')
np_64_2 = read_raw_data(filename='JET.64')

new_64 = np.zeros(shape=(64,64))
new_64 = np.uint8(np_64)

# 對每一像素做平均
for i in range(63):
    for j in range(63):
        new_64[i][j] = (np_64_1[i][j] + np_64_2[i][j]) / 2

plot_2_subplot(img=new_64, title='Task: Average image of LINCON and JET images')


#===========  part 2-4 =====================
np_64 = read_raw_data(filename='LINCOLN.64')

# 先向右平移
right = np.zeros(shape=(64,64))
right = np.uint8(right)

right[::, 1:] = np_64[::, 0:-1]
right[::,0]   = 31 # 填滿空白


sub_result = np.zeros(shape=(64,64))
sub_result = np.uint8(sub_result)

# 原始圖像 - 右移圖像 = 淺去左方像素資料
for i in range(63):
    for j in range(63):
        if (int(np_64[i][j]) - int(right[i][j])) <= 0: # 設定下限，轉為int才不會溢位
            sub_result[i][j] = 0

        elif (int(np_64[i][j]) - int(right[i][j])) >= 31: # 設定上限
            sub_result[i][j] = 31

        else:
            sub_result[i][j] = int(np_64[i][j]) - int(right[i][j])

plot_2_subplot(img=right, title='Task: shift right 1 pixel')
plot_2_subplot(img=sub_result, title='Task: g(x,y) = f(x,y) - f(x-1,y)')