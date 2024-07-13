import numpy as np
import cv2
from sklearn.cluster import MeanShift, estimate_bandwidth

# 生成样本数据
img = cv2.imread('./lena.jpg', 0)
pixel = img.ravel()
pixel_list = np.bincount(pixel, minlength=256)
# 使用Mean Shift算法聚类
bandwidth = estimate_bandwidth(pixel_list.reshape(-1, 1), quantile=0.2)
ms = MeanShift(bandwidth=70, bin_seeding=True)
ms.fit(pixel_list.reshape(-1, 1))
labels = ms.labels_

# 打印每个样本所属的类别
print("聚类标签:", labels)
# 获取聚类中心
cluster_centers = ms.cluster_centers_
n_clusters_ = len(cluster_centers)
print("Number of estimated clusters : %d" % n_clusters_)
print("Cluster centers : \n", cluster_centers)