import tensorflow as tf
from keras import layers, models
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class RBFLayer(layers.Layer):
    def __init__(self, units, gamma, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.gamma = gamma

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers',
                                       shape=(self.units, input_shape[1]),
                                       initializer='uniform',
                                       trainable=True)
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        diff = tf.expand_dims(inputs, axis=1) - self.centers
        norm = tf.norm(diff, axis=-1)
        return tf.exp(-self.gamma * tf.square(norm))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)


# Load MNIST dataset
# 读取MAT文件
mat_data = scipy.io.loadmat('landsat.mat')
variable_value = mat_data['data']
variable_label = mat_data['label']
ss = StandardScaler()
ss.fit(variable_value)
value_ss = ss.transform(variable_value)
x_train, x_test, y_train, y_test = train_test_split(value_ss,variable_label,train_size=0.75,test_size=0.25,random_state=42,shuffle=True)
y_train = y_train - 1
y_test = y_test -1


num_centers = 1000
gamma = 0.1
# Build RBFNN model
model = models.Sequential([
    layers.Flatten(input_shape=(36,)),
    RBFLayer(units=num_centers, gamma=gamma),
    layers.Dense(6, activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 打印模型概况
model.summary()

# Train the model
model.fit(x_train, y_train, epochs=100)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')

with open('rbf.txt', 'a') as file:
    file.write('Centers: {}\n'.format(num_centers))
    file.write('Gamma: {}\n'.format(gamma))
    file.write('准确率: {:.4f}\n'.format(test_acc))
