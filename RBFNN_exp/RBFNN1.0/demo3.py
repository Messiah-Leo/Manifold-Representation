import numpy as np


class RBFNN:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.001, num_epochs=100):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        # 初始化隐藏层参数
        self.centers = np.random.rand(hidden_dim, input_dim)
        # self.centers = [[0.1], [0.3], [0.5], [0.7], [0.9]]
        self.sigmas = np.random.rand(hidden_dim)
        self.weights = np.random.rand(output_dim, hidden_dim)
        self.biases = np.random.rand(output_dim)

    @staticmethod
    def radial_basis_function(x, center, sigma):
        return np.exp(-np.linalg.norm(x - center) ** 2 / (2 * sigma ** 2))

    def hidden_layer(self, x):
        hidden_output = np.zeros(self.hidden_dim)
        for i in range(self.hidden_dim):
            hidden_output[i] = self.radial_basis_function(x, self.centers[i], self.sigmas[i])
        return hidden_output

    def output_layer(self, hidden_output):
        return np.dot(self.weights, hidden_output) + self.biases

    def loss(self, predicted, target):
        return np.mean((predicted - target) ** 2)

    def train(self, X, y):
        for epoch in range(self.num_epochs):
            total_loss = 0
            for i in range(len(X)):
                x = X[i]
                target = y[i]

                # 前向传播
                hidden_output = self.hidden_layer(x)
                predicted = self.output_layer(hidden_output)

                # 计算损失
                loss = self.loss(predicted, target)
                total_loss += loss

                # 反向传播和权重更新
                error = predicted - target
                delta_weights = np.outer(error, hidden_output)
                delta_biases = error
                delta_hidden = np.dot(error, self.weights)

                delta_sigma = np.zeros(self.hidden_dim)
                for j in range(self.hidden_dim):
                    delta_sigma[j] = np.sum((error * self.weights[:, j] * (x - self.centers[j]))) / (self.sigmas[j] ** 3)

                self.weights -= self.learning_rate * delta_weights
                self.biases -= self.learning_rate * delta_biases
                self.centers -= self.learning_rate * delta_sigma[:, np.newaxis] * (x - self.centers)
                self.sigmas -= self.learning_rate * delta_sigma * (np.linalg.norm(x - self.centers, axis=1) ** 2)

            avg_loss = total_loss / len(X)
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {avg_loss:.4f}")
            if avg_loss < 0.5:
                break

    def predict(self, X):
        predictions = []
        for x in X:
            hidden_output = self.hidden_layer(x)
            predicted = self.output_layer(hidden_output)
            predictions.append(predicted)
        return np.array(predictions)


# 示例数据
X = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13]])
y = np.array([[0], [2], [4], [6], [8], [10], [12], [14], [16], [18], [20], [22], [24], [26]])

# 创建并训练RBF神经网络
input_dim = 1
hidden_dim = 500
output_dim = 1
rbfnn = RBFNN(input_dim, hidden_dim, output_dim)
rbfnn.train(X, y)

# 进行预测
X_test = np.array([[2], [20]])
predictions = rbfnn.predict(X_test)
print("Predictions:", predictions)