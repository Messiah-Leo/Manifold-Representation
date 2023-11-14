from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import joblib

class RBFNN(BaseEstimator, RegressorMixin):
    def __init__(self, num_centers=10, gamma=1.0, alpha=1e-6):
        self.num_centers = num_centers
        self.gamma = gamma
        self.alpha = alpha
        self.centers = None
        self.weights = None

    def fit(self, X, y):
        # 使用K-means选择RBF中心
        kmeans = KMeans(n_clusters=self.num_centers, n_init='auto')
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_

        # 计算设计矩阵
        design_matrix = self._radial_basis(X, self.centers, self.gamma)

        # 使用岭回归拟合权重
        ridge = Ridge(alpha=self.alpha)
        ridge.fit(design_matrix, y)
        self.weights = ridge.coef_

    def predict(self, X):
        design_matrix = self._radial_basis(X, self.centers, self.gamma)
        return np.dot(design_matrix, self.weights)

    def _radial_basis(self, X, centers, gamma):
        dist = euclidean_distances(X, centers)
        return np.exp(-gamma * dist ** 2)


# 加载MNIST数据集
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X, y = mnist['data'], mnist['target']

# 数据预处理
X = StandardScaler().fit_transform(X)
y = y.astype(int)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建RBF神经网络
centers = np.linspace(10, 100, 10)
gamma = 3
for num_centers in centers:
    num_centers = int(num_centers)
    print(f"Calculating rbf_model: centers_{num_centers} gamma_{gamma}")
    rbf = RBFNN(num_centers=num_centers, gamma=gamma, alpha=1e-1)

    # 训练模型
    rbf.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = rbf.predict(X_test)

    # 计算准确率
    accuracy = np.mean(y_pred == y_test)
    print("准确率:", accuracy)

    with open('results.txt', 'a') as file:
        file.write('Centers: {}\n'.format(num_centers))
        file.write('Gamma: {}\n'.format(gamma))
        file.write('准确率: {:.4f}\n'.format(accuracy))
    # 保存模型
    model_name = f"rbf_model_centers_{num_centers}_gamma_{gamma}.pkl"
    joblib.dump(rbf, model_name)