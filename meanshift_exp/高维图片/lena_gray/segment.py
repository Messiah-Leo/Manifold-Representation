import numpy as np
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
from PIL import Image

# 加载 Lena 灰度图像
lena_image = Image.open('lena_gray.png')
lena_array = np.array(lena_image)

# 将图像转换为一维数组
height, width = lena_array.shape
data = lena_array.reshape(height * width, 1)
bws = np.linspace(20, 25, 6)
for bw in bws:
    # 使用 MeanShift 进行聚类
    meanshift = MeanShift(bandwidth=bw, n_jobs=-1)  # 根据需要调整带宽
    meanshift.fit(data)

    # 提取聚类结果
    labels = meanshift.labels_
    cluster_centers = meanshift.cluster_centers_

    # 将聚类结果重新映射回图像尺寸
    segmented_image = cluster_centers[labels].reshape(height, width)

    plt.imshow(segmented_image, cmap='gray')
    plt.axis('off')
    plt.savefig(f'image/B{bw}.png', dpi=1000, bbox_inches='tight', pad_inches=0)
