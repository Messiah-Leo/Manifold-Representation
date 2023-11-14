#!/user/bin/env python
# _*_ coding: utf-8 _*_
# @Time: 2023/10/20 17:05
# @Author: X.D.
# @Version: V 0.1
# @File: layers
"""
采用前馈RBFNN网络，由meanshift构建子网络
"""

import numpy as np
import time


class RBFLayer(object):
    def __init__(self, num_input, num_hidden, num_output):  # 全连接层初始化
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output
        print('\tConnected layer with input %d, hidden %d, output %d.' % (self.num_input, self.num_hidden, self.num_output))

    def init_param(self, std=0.01):  # 参数初始化
        self.centers = np.random.rand(self.num_hidden, self.num_input)
        self.betas = np.random.rand(self.num_hidden)  # 不同节点用不同的带宽，感知不同范围的响应
        self.weight = np.random.normal(loc=0.0, scale=std, size=(self.num_output, self.num_hidden))
        self.bias = np.zeros(self.num_output)

    @staticmethod
    def radial_basis_function(x, center, beta):
        return np.exp(-beta * np.linalg.norm(x - center) ** 2)

    def forward(self, input):  # 前向传播，计算输出结果
        start_time = time.time()
        self.input = input
        self.hidden_output = np.zeros(self.num_hidden)
        for i in range(self.num_hidden):
            # TODO: 隐藏层的运算还可以进行优化（10.18）
            self.hidden_output[i] = self.radial_basis_function(self.input, self.centers[i], self.betas[i])
        self.output = np.matmul(self.weight, self.hidden_output) + self.bias
        return self.output

    def backward(self, top_diff):  # 反向传播，计算参数梯度和本层损失
        self.d_weight = np.dot(top_diff.T, self.hidden_output.T)
        self.d_bias = np.sum(top_diff, axis=0)
        self.d_betas = np.dot(top_diff.T, self.hidden_output.T)
        bottom_diff = np.dot(self.weight.T, top_diff.T)
        return bottom_diff

    def update_param(self, lr):  # 参数更新
        # TODO: 目前进度
        self.weight = self.weight - lr * self.d_weight
        self.bias = self.bias - lr * self.d_bias

    def load_param(self, centers, betas, weight, bias):  # 参数加载
        assert self.centers.shape == centers.shape
        assert self.betas.shape == betas.shape
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.centers = centers
        self.betas = betas
        self.weight = weight
        self.bias = bias

    def save_param(self):  # 参数保存
        return self.centers, self.betas, self.weight, self.bias


class SoftmaxLossLayer(object):
    def __init__(self):
        print('\tSoftmax loss layer.')

    def forward(self, input):  # 损失层的前向传播
        input_max = np.max(input, axis=1, keepdims=True)
        input_exp = np.exp(input - input_max)
        self.prob = input_exp / np.sum(input_exp, axis=1, keepdims=True)
        return self.prob

    def get_loss(self, label):  # 计算损失
        self.batch_size = self.prob.shape[0]
        self.label_onehot = np.zeros_like(self.prob)
        self.label_onehot[np.arange(self.batch_size), label] = 1.0
        loss = -np.sum(np.log(self.prob) * self.label_onehot) / self.batch_size
        return loss

    def backward(self):  # 损失层的反向传播
        bottom_diff = (self.prob - self.label_onehot) / self.batch_size
        return bottom_diff
