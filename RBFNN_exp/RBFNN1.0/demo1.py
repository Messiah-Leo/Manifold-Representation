import numpy as np

class RBFNN:
    def __init__(self, num_hidden, num_outputs):
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.centers = None
        self.widths = None
        self.weights = None

    def initialize_centers(self, inputs):
        indices = np.random.choice(len(inputs), self.num_hidden, replace=False)
        self.centers = inputs[indices]

    def initialize_widths(self, inputs):
        max_distance = np.max(np.linalg.norm(self.centers[:, np.newaxis] - inputs, axis=2))
        self.widths = np.repeat(max_distance / np.sqrt(2 * self.num_hidden), self.num_hidden)

    def gaussian(self, x, center, width):
        return np.exp(-np.linalg.norm(x - center) ** 2 / (2 * width ** 2))

    def calculate_activations(self, inputs):
        activations = np.zeros((len(inputs), self.num_hidden))
        for i, x in enumerate(inputs):
            for j in range(self.num_hidden):
                activations[i, j] = self.gaussian(x, self.centers[j], self.widths[j])
        return activations

    def fit(self, inputs, targets, learning_rate=0.01, epochs=100):
        self.initialize_centers(inputs)
        self.initialize_widths(inputs)
        self.weights = np.random.randn(self.num_hidden, self.num_outputs)

        for _ in range(epochs):
            activations = self.calculate_activations(inputs)
            outputs = np.dot(activations, self.weights)
            errors = targets - outputs

            delta_weights = learning_rate * np.dot(activations.T, errors)
            self.weights += delta_weights

    def predict(self, inputs):
        activations = self.calculate_activations(inputs)
        outputs = np.dot(activations, self.weights)
        return outputs

# 示例用法
inputs = np.linspace(-5, 5, 100).reshape(-1, 1)
targets = np.sin(inputs) + np.random.normal(0, 0.1, size=inputs.shape)

rbfnn = RBFNN(num_hidden=10, num_outputs=1)
rbfnn.fit(inputs, targets, learning_rate=0.01, epochs=1000)

predictions = rbfnn.predict(inputs)

# 绘制拟合曲线
import matplotlib.pyplot as plt

plt.scatter(inputs, targets, label='True')
plt.plot(inputs, predictions, color='red', label='Predicted')
plt.legend()
plt.show()