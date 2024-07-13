import tensorflow as tf
from keras import layers, models
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.cluster import MeanShift
from sklearn.preprocessing import StandardScaler

# 读取MAT文件
mat_data = scipy.io.loadmat('dna.mat')
variable_value = mat_data['data']
variable_label = mat_data['label']
ss = StandardScaler()
ss.fit(variable_value)
value_ss = ss.transform(variable_value)
x_train, x_test, y_train, y_test = train_test_split(value_ss,variable_label,train_size=0.75,test_size=0.25,random_state=42,shuffle=True)
y_train = y_train - 1
y_test = y_test -1

bandwidth = 10
# 使用 Mean Shift 算法确定中心
meanshift = MeanShift(bandwidth=bandwidth)
meanshift.fit(x_train)
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
                                       trainable=True)  # 固定中心
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
    RBFLayer(units=len(centers), gamma=gamma, center_initializer=centers, input_shape=(180,)),
    layers.Dense(3, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 打印模型概况
model.summary()

# Train the model
model.fit(x_train, y_train, epochs=30)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')

with open('results.txt', 'a') as file:
    file.write('Bandwidth: {}; '.format(bandwidth))
    file.write('Centers: {}; '.format(len(centers)))
    file.write('Gamma: {}; '.format(gamma))
    file.write('准确率: {:.4f}\n'.format(test_acc))
