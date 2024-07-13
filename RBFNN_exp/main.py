from RandomImage import random_image
import scipy.io
import tensorflow as tf
from keras import layers, models
from sklearn.cluster import MeanShift

# 读取MAT文件
mat_data = scipy.io.loadmat('./数据汇总/Segment.mat')
variable_value = mat_data['data']
variable_label = mat_data['label']
x_train, x_test, ID_train, ID_test = random_image(variable_value.T, 330, 115, 7, 1)
y_train = variable_label[ID_train] - 1
y_test = variable_label[ID_test] - 1

bandwidth = 0.1
# 使用 Mean Shift 算法确定中心
meanshift = MeanShift(bandwidth=bandwidth)
meanshift.fit(x_test)
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

    def call(self, inputs, **kwargs):
        diff = tf.expand_dims(inputs, axis=1) - self.centers
        norm = tf.norm(diff, axis=-1)
        return tf.exp(-self.gamma * tf.square(norm))

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units


# 构建 RBFNN 模型
gamma =  1000
model = models.Sequential([
    RBFLayer(units=len(centers), gamma=gamma, center_initializer=centers, input_shape=(18,)),
    layers.Dense(7, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 打印模型概况
model.summary()

# 训练模型
model.fit(x_train / 255.0, y_train, epochs=5)  # 数据归一化

# Evaluate the model
x_test_flatten = x_test.reshape(x_test.shape[0], -1)
test_loss, test_acc = model.evaluate(x_test_flatten / 255.0, y_test)
print(f'Test accuracy: {test_acc}')

with open('results.txt', 'a') as file:
    file.write('Bandwidth: {}; '.format(bandwidth))
    file.write('Centers: {}; '.format(len(centers)))
    file.write('Gamma: {}; '.format(gamma))
    file.write('准确率: {:.4f}\n'.format(test_acc))
