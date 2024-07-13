import pandas as pd
import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.cluster import MeanShift
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from scipy.sparse.csgraph import connected_components
from scipy.sparse.csgraph import shortest_path
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt

# 读取MAT文件
mat_data = scipy.io.loadmat('Segment.mat')
variable_value = mat_data['data']
variable_label = mat_data['label']
ss = StandardScaler()
ss.fit(variable_value)
value_ss = ss.transform(variable_value)
x_train, x_test, y_train, y_test = train_test_split(value_ss, variable_label, train_size=0.4, test_size=0.6,
                                                    random_state=42, shuffle=True)
y_train = y_train - 1
y_test = y_test - 1

bandwidth = 0.5
# 使用 Mean Shift 算法确定中心
ms1 = MeanShift(bandwidth=bandwidth)
ms1.fit(x_train)
label1 = pd.DataFrame(ms1.labels_, columns=['Cluster ID'])
data_c1 = ms1.cluster_centers_

# 得到子集的标签
cluster_ids = label1['Cluster ID'].unique()
# 排序后的标签
sorted_cluster_ids = sorted(cluster_ids)

t = pd.DataFrame(y_train, index=ms1.labels_)
t1 = list()
for i in sorted_cluster_ids:
    t1.append(np.unique(t.loc[i]))
# 计算子集之间的欧氏距离
sorted_distances = pairwise_distances(x_train, metric='euclidean')
distance_matrix = pd.DataFrame(sorted_distances, index=label1['Cluster ID'], columns=label1['Cluster ID'])
dismatrix = pd.DataFrame(index=sorted_cluster_ids, columns=sorted_cluster_ids)
for i in sorted_cluster_ids:
    for j in sorted_cluster_ids:
        if i != j:
            dismatrix.loc[i, j]=distance_matrix.loc[i, j].min().min()
        else:
            dismatrix.loc[i, j]=99
distance_matrix_np = np.array(dismatrix)

# 根据阈值判断近邻
for i in range(distance_matrix_np.shape[0]):
    sorted_distances = np.sort(distance_matrix_np[i])
    # Set all distances after the thread distance to zero
    distance_matrix_np[i][distance_matrix_np[i] > 2] = 0
    distance_matrix_np[i][distance_matrix_np[i] != 0] = 1

# 将相邻的子集合并并打上标签
n_connected_components, labels = connected_components(distance_matrix_np.astype(np.float64))
distance_matrix_np = distance_matrix_np[labels == 0, :]
distance_matrix_np = distance_matrix_np[:, labels == 0]

c1 = data_c1[labels == 0, :]

# ISOMAP降维
# dis_matrix = pairwise_distances(c1, metric='euclidean')
# M = np.multiply(dis_matrix, distance_matrix_np).astype(np.float64)
# dist_matrix_ = shortest_path(M, method="auto", directed=False)
# G = dist_matrix_**2
# G *= -0.5
# transformer = KernelPCA(n_components=2, kernel="precomputed",eigen_solver="auto")
# X_transformed = transformer.fit_transform(G)
# # plot the isomap X_transformedection
# plt.scatter(X_transformed[:, 0], X_transformed[:, 1], s=200, lw=0, alpha=1)
# # Iterate over each row and column in the distance_matrix_np array
# for i in range(distance_matrix_np.shape[0]):
#     for j in range(distance_matrix_np.shape[1]):
#         # Check if the value is non-zero
#         if distance_matrix_np[i][j] != 0:
#             # Connect the points in data_c1 corresponding to the cluster IDs
#             plt.plot([X_transformed[i][0], X_transformed[j][0]], [X_transformed[i][1], X_transformed[j][1]],'blue')
#
# # Plot the scatter plot of data_c1 with cluster IDs
# # plt.scatter(x=result1['x'], y=result1['y'], c=result1['Cluster ID'], s=5)
#
# # Show the plot
# plt.show()
