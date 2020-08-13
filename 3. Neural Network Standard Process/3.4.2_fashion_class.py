# 六步法之class搭建神经网络
# 用MNIST中FASHION数据集识别衣裤
# mengld
# 2020-8-13

# import
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model

# train, test
fashion = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# model
class MnistModel(Model):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.flatten(x)
        x = self.d1(x)
        y = self.d2(x)
        return y

model = MnistModel()
# model.compile
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics = ['sparse_categorical_accuracy'])

# model.fit
model.fit(x_train, y_train, batch_size = 32, epochs=5, validation_data=(x_test, y_test), validation_freq=1)
model.summary()

# model.summary
