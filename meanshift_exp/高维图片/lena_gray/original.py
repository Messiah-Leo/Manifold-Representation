import numpy as np
from PIL import Image

# 加载 Lena 灰度图像
lena_image = Image.open('lena_gray.png')
lena_array = np.array(lena_image)

def mean_shift(data, bandwidth):
    centroids = data
    count = 0
    while True:
        print(f'第{count}次迭代')
        distances = centroids[:, np.newaxis] - centroids
        distances = np.triu(np.sum(np.square(distances), axis=2))
        weights = np.exp(-0.5 * (distances / bandwidth) ** 2)
        sum_weights = np.sum(weights, axis=0)
        new_centroids = np.dot(weights, centroids) / sum_weights[:, np.newaxis]
        if np.allclose(new_centroids, centroids, atol=1e-9):
            break
        centroids = new_centroids
        count += 1
    return np.round(centroids, 3)


# 加载 Lena 灰度图像
lena_image = Image.open('lena_gray.png')
lena_array = np.array(lena_image)
bandwidth = 30

cluster_centers = mean_shift(data.values, bandwidth)
# 获取聚类中心
cluster_center = np.unique(cluster_centers)
n_clusters_ = len(cluster_center)
center = cluster_center.tolist()
print("Number of estimated clusters: %d" % n_clusters_)
print("Cluster centers:\n", center)