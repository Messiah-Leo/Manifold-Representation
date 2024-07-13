import cv2
import numpy as np

# 加载图像
image = cv2.imread('lena.jpg')

# 转换为Lab颜色空间
lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# 提取a和b通道
ab_channels = lab_image[:, :, 1:].astype(np.float32)

# Reshape为二维数组
ab_channels_flat = ab_channels.reshape((-1, 2))

# MeanShift聚类
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
_, labels, centers = cv2.meanShift(ab_channels_flat, None, criteria)

# 将每个像素的标签转换为uint8类型
labels = labels.astype(np.uint8)

# 将每个标签转换为对应的颜色
segmented_image = np.zeros_like(ab_channels)
for i, center in enumerate(centers):
    segmented_image[labels == i] = center

# 转换回BGR颜色空间
segmented_image = cv2.cvtColor(segmented_image.astype(np.uint8), cv2.COLOR_LAB2BGR)

# 显示原始图像和分割结果
cv2.imshow('Original Image', image)
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()