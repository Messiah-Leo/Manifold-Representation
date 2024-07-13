import cv2
import numpy as np
import xlwt
from scipy.interpolate import interp1d

img = cv2.imread('./lena.jpg', 0)
pixel = img.ravel()
pixel_list = np.bincount(pixel, minlength=256)

# 创建插值函数
x = np.arange(256)
y = pixel_list
f = interp1d(x, y, kind='cubic')

# 在更密集的x范围内进行插值
x_new = np.linspace(0, 255, 500)
y_new = f(x_new)

# 将插值结果存储到data中
data = np.column_stack((x_new, y_new))

# 去除值为零的点
data_filtered = data[data[:, 1] >= 0.0001]
'''
data=[]
for i in range(256):
    if pixel_list[i]:
        data.append((i, pixel_list[i]))
data_filtered = np.array(data)
'''
book = xlwt.Workbook(encoding='utf-8', style_compression=0)
sheet = book.add_sheet('一维meanshift', cell_overwrite_ok=True)
col = np.linspace(5, 30, 50)
for i in range(len(col)):
    sheet.write(0, i, col[i])


def mean_shift(pixel_list, bandwidth):
    data = pixel_list[:, 0]
    centroids = np.copy(data)
    while True:
        distances = np.abs(centroids - centroids[:, np.newaxis])
        weights = np.exp(-0.5 * (distances / bandwidth) ** 2)
        new_centroids = np.sum(pixel_list[:, 1] * centroids[:, np.newaxis] * weights, axis=0) / np.sum(
            pixel_list[:, 1] * weights, axis=0)
        if np.all(np.isclose(new_centroids, centroids, atol=1e-9)):
            break
        centroids = new_centroids
    return np.round(centroids, 3)


bands = col
for count, bandwidth in zip(range(len(bands)), bands):
    print(count)
    cluster_centers = mean_shift(data_filtered, bandwidth)
    # 获取聚类中心
    cluster_center = np.unique(cluster_centers)
    center = list(cluster_center)

    labels = []
    k = 0
    for _ in data_filtered:
        labels.append(center.index(cluster_centers[k]))
        k = k + 1

    i = 1
    for label in labels:
        sheet.write(i, count, label)
        i = i + 1

book.save('C:/Users/XD/Desktop/MeanShift/一维/lena/C25.xls')
