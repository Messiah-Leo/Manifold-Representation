#!/user/bin/env python
# _*_ coding: utf-8 _*_
# @Time: 2023/11/15 15:08
# @Author: X.D.
# @Version: V 0.3
# @File: main.py

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.cluster import MeanShift
from sklearn.preprocessing import StandardScaler
import numpy as np

# 加载 MNIST 数据集
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 数据预处理
x_train_flatten = x_train.reshape(x_train.shape[0], -1)

# 根据每个类别选择500个样本
selected_indices = []
selected_number = 300
for i in range(10):  # 10类
    indices = np.where(y_train == i)[0][:selected_number]
    selected_indices.extend(indices)

x_train_selected = x_train_flatten[selected_indices]
y_train_selected = y_train[selected_indices]

# 标准化数据
scaler = StandardScaler()
x_train_selected = scaler.fit_transform(x_train_selected)

bandwidth = 15
# 使用 Mean Shift 算法确定中心
meanshift = MeanShift(bandwidth=bandwidth)
meanshift.fit(x_train_selected)
centers = meanshift.cluster_centers_

# 定义 RBFLayer，使用 Mean Shift 确定的中心进行初始化
class RBFLayer(layers.Layer):
    def __init__(self, units, gamma, center_initializer, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.gamma = gamma
        self.center_initializer = center_initializer

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers',
                                       shape=(self.units, input_shape[1]),
                                       initializer=tf.constant_initializer(self.center_initializer),
                                       trainable=False)  # 固定中心
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        diff = tf.expand_dims(inputs, axis=1) - self.centers
        norm = tf.norm(diff, axis=-1)
        return tf.exp(-self.gamma * tf.square(norm))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)


# 构建 RBFNN 模型
gamma = 0.01
model = models.Sequential([
    RBFLayer(units=len(centers), gamma=gamma, center_initializer=centers, input_shape=(784,)),
    layers.Dense(10, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 打印模型概况
model.summary()

# 训练模型
model.fit(x_train_flatten / 255.0, y_train, epochs=5)  # 数据归一化

# Evaluate the model
x_test_flatten = x_test.reshape(x_test.shape[0], -1)
test_loss, test_acc = model.evaluate(x_test_flatten / 255.0, y_test)
print(f'Test accuracy: {test_acc}')

with open('results.txt', 'a') as file:
    file.write('Bandwidth: {}; '.format(bandwidth))
    file.write('Centers: {}; '.format(len(centers)))
    file.write('Gamma: {}; '.format(gamma))
    file.write('准确率: {:.4f}\n'.format(test_acc))
