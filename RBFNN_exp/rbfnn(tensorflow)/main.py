import tensorflow as tf
from tensorflow.keras import layers, models


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
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Data preprocessing
x_train, x_test = x_train / 255.0, x_test / 255.0
num_centers = 300
gamma = 0.03
# Build RBFNN model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    RBFLayer(units=num_centers, gamma=gamma),
    layers.Dense(10, activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 打印模型概况
model.summary()

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')

with open('results.txt', 'a') as file:
    file.write('Centers: {}\n'.format(num_centers))
    file.write('Gamma: {}\n'.format(gamma))
    file.write('准确率: {:.4f}\n'.format(test_acc))
