import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def gaussian_kernel(distance, bandwidth):
    return (1 / (bandwidth * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((distance / bandwidth)) ** 2)


def mean_shift(points, bandwidth=1, min_distance=1e-4):
    shifted_points = np.array(points)
    num_points = len(points)

    while True:
        new_points = np.copy(shifted_points)

        def shift_point(i):
            point = shifted_points[i]
            new_point = np.zeros_like(point)
            total_weight = 0.0
            for j in range(num_points):
                distance = euclidean_distance(point, shifted_points[j])
                weight = gaussian_kernel(distance, bandwidth)
                new_point += shifted_points[j] * weight
                total_weight += weight

            new_point /= total_weight
            return new_point

        # 并行计算点的移动
        with Parallel(n_jobs=-1) as parallel:
            new_points = parallel(delayed(shift_point)(i) for i in range(num_points))

        # 检查点的移动距离是否小于阈值
        shift_distances = [euclidean_distance(shifted_points[i], new_points[i]) for i in range(num_points)]
        max_shift_distance = max(shift_distances)
        if max_shift_distance < min_distance:
            break

        shifted_points = np.array(new_points)

    return shifted_points


# 示例数据点
data = pd.read_csv('data1014.csv', header=None)
bandwidth = 30

cluster_centers = mean_shift(data.values, bandwidth)
# 获取聚类中心
cluster_center = np.unique(cluster_centers)
n_clusters_ = len(cluster_center)
center = cluster_center.tolist()
print("Number of estimated clusters: %d" % n_clusters_)
print("Cluster centers:\n", center)
