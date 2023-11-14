#!/user/bin/env python
# _*_ coding: utf-8 _*_
# @Time: 2023/10/18 11:40
# @Author: X.D.
# @Version: V 0.2
# @File: Demo

import numpy as np
import struct
import os

MNIST_DIR = "./mnist_data"
TRAIN_DATA = "train-images-idx3-ubyte"
TRAIN_LABEL = "train-labels-idx1-ubyte"
TEST_DATA = "t10k-images-idx3-ubyte"
TEST_LABEL = "t10k-labels-idx1-ubyte"


class RBFNN:
    def __init__(self, batch_size=200, input_dim=784, hidden_dim=64, output_dim=10, lr=0.01, num_epochs=100,
                 print_iter=100):
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = lr
        self.num_epochs = num_epochs
        self.print_iter = print_iter

        # 初始化隐藏层参数
        self.centers = np.random.rand(hidden_dim, input_dim)
        self.sigmas = np.random.rand(hidden_dim)
        self.weights = np.random.rand(output_dim, hidden_dim)
        self.biases = np.random.rand(output_dim)

    @staticmethod
    def load_mnist(file_dir, is_images):
        # Read binary data
        bin_file = open(file_dir, 'rb')
        bin_data = bin_file.read()
        bin_file.close()
        # Analysis file header
        if is_images:
            # Read images
            fmt_header = '>iiii'
            magic, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, 0)
        else:
            # Read labels
            fmt_header = '>ii'
            magic, num_images = struct.unpack_from(fmt_header, bin_data, 0)
            num_rows, num_cols = 1, 1
        data_size = num_images * num_rows * num_cols
        mat_data = struct.unpack_from('>' + str(data_size) + 'B', bin_data, struct.calcsize(fmt_header))
        mat_data = np.reshape(mat_data, [num_images, num_rows * num_cols])
        print('Load images from %s, number: %d, data shape: %s' % (file_dir, num_images, str(mat_data.shape)))
        return mat_data

    def load_data(self):
        # TODO: 调用函数 load_mnist 读取和预处理 MNIST 中训练数据和测试数据的图像和标记
        print('Loading MNIST data from files...')
        train_images = self.load_mnist(os.path.join(MNIST_DIR, TRAIN_DATA), True)
        train_labels = self.load_mnist(os.path.join(MNIST_DIR, TRAIN_LABEL), False)
        test_images = self.load_mnist(os.path.join(MNIST_DIR, TEST_DATA), True)
        test_labels = self.load_mnist(os.path.join(MNIST_DIR, TEST_LABEL), False)
        self.train_data = np.append(train_images, train_labels, axis=1)
        self.test_data = np.append(test_images, test_labels, axis=1)

    def shuffle_data(self):
        print('Randomly shuffle MNIST data...')
        np.random.shuffle(self.train_data)

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

    @staticmethod
    def loss(predicted, target):
        return np.mean((predicted - target) ** 2)

    def train(self):
        max_batch = int(self.train_data.shape[0] / self.batch_size)
        for idx_epoch in range(self.num_epochs):
            total_loss = 0
            for idx_batch in range(self.num_epochs):
                batch_images = self.train_data[idx_batch * self.batch_size:(idx_batch + 1) * self.batch_size, :-1]
                batch_labels = self.train_data[idx_batch * self.batch_size:(idx_batch + 1) * self.batch_size, -1]

                # 前向传播
                hidden_output = self.hidden_layer(batch_images)
                predicted = self.output_layer(hidden_output)

                # 计算损失
                loss = self.loss(predicted, batch_labels)
                total_loss += loss

                # 反向传播和权重更新
                error = predicted - batch_labels
                delta_weights = np.outer(error, hidden_output)
                delta_biases = error
                delta_hidden = np.dot(error, self.weights)

                for j in range(self.hidden_dim):
                    delta_sigma = (error * hidden_output[j] * (batch_labels - self.centers[j])) / (self.sigmas[j] ** 3)
                    self.weights -= self.lr * delta_weights
                    self.biases -= self.lr * delta_biases
                    self.centers[j] -= self.lr * delta_sigma
                    self.sigmas[j] -= self.lr * delta_sigma[0]

                if idx_batch % self.print_iter == 0:
                    print('Epoch %d, iter %d, loss: %.6f' % (idx_epoch, idx_batch, loss))


if __name__ == "__main__":
    # 创建并训练RBF神经网络
    rbfnn = RBFNN(hidden_dim=128)
    rbfnn.load_data()
    rbfnn.train()
