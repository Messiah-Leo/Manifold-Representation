import pandas as pd
import numpy as np
import scipy.io
from sklearn.datasets import make_swiss_roll
from sklearn.cluster import MeanShift
from sklearn.metrics import pairwise_distances
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse.csgraph import connected_components
from sklearn.utils.graph import _fix_connected_components
from scipy.sparse.csgraph import shortest_path
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt

X, t = make_swiss_roll(n_samples=500, noise=0.2)

bandwidth = 2
# 使用 Mean Shift 算法确定中心
ms1 = MeanShift(bandwidth=bandwidth)
ms1.fit(X)
label1 = pd.DataFrame(ms1.labels_, columns=['Cluster ID'])
data_c1 = ms1.cluster_centers_
# Get the Cluster IDs from the result1 dataframe
cluster_ids = label1['Cluster ID'].unique()

# Sort the cluster IDs in ascending order
sorted_cluster_ids = sorted(cluster_ids)

t = pd.DataFrame(t, index=ms1.labels_)
t1 = np.zeros(len(cluster_ids))
for i in sorted_cluster_ids:
    t1[i] = np.mean(t.loc[i])
# Calculate the pairwise distances between the sorted cluster IDs
sorted_distances = pairwise_distances(X, metric='euclidean')

# Create a dataframe with sorted Cluster IDs as rows and columns
distance_matrix = pd.DataFrame(sorted_distances, index=label1['Cluster ID'], columns=label1['Cluster ID'])
dismatrix = pd.DataFrame(index=sorted_cluster_ids, columns=sorted_cluster_ids)
for i in sorted_cluster_ids:
    for j in sorted_cluster_ids:
        if i != j:
            dismatrix.loc[i, j] = np.min(np.min(distance_matrix.loc[i, j]))
        else:
            dismatrix.loc[i, j] = 99

# Convert the distance_matrix dataframe to a numpy array
distance_matrix_np = np.array(dismatrix)

# Iterate over each row in the distance_matrix_np array
for i in range(distance_matrix_np.shape[0]):
    # Sort the distances in the current row in ascending order
    sorted_distances = np.sort(distance_matrix_np[i])
    # print(sorted_distances[1])

    # Set all distances after the thread distance to zero
    distance_matrix_np[i][distance_matrix_np[i] > 5.3] = 0
    distance_matrix_np[i][distance_matrix_np[i] != 0] = 1
"""
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.scatter(data_c1[:, 0], data_c1[:, 1], data_c1[:, 2], c=t1)
for i in range(distance_matrix_np.shape[0]):
    for j in range(distance_matrix_np.shape[1]):
        if distance_matrix_np[i][j] != 0:
            plt.plot([data_c1[i][0], data_c1[j][0]], [data_c1[i][1], data_c1[j][1]], [data_c1[i][2], data_c1[j][2]],
                     'blue')
plt.show()

"""
dis_matrix = pairwise_distances(data_c1, metric='euclidean')
M = np.multiply(dis_matrix, distance_matrix_np).astype(np.float64)
dist_matrix_ = shortest_path(M, method="auto", directed=False)
G = dist_matrix_ ** 2
G *= -0.5
transformer = KernelPCA(n_components=2, kernel="precomputed", eigen_solver="auto")
X_transformed = transformer.fit_transform(G)
# plot the isomap X_transformedection
plt.scatter(X_transformed[:, 0], X_transformed[:, 1], s=200, lw=0, alpha=1,c=t1)
# Iterate over each row and column in the distance_matrix_np array
for i in range(distance_matrix_np.shape[0]):
    for j in range(distance_matrix_np.shape[1]):
        # Check if the value is non-zero
        if distance_matrix_np[i][j] != 0:
            # Connect the points in data_c1 corresponding to the cluster IDs
            plt.plot([X_transformed[i][0], X_transformed[j][0]], [X_transformed[i][1], X_transformed[j][1]], 'blue')

# Plot the scatter plot of data_c1 with cluster IDs
# plt.scatter(x=result1['x'], y=result1['y'], c=result1['Cluster ID'], s=5)

# Show the plot
plt.show()

