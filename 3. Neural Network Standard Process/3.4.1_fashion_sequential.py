# 六步法之sequential搭建神经网络
# 用MNIST中FASHION数据集识别衣裤
# mengld
# 2020-8-13

# import
import tensorflow as tf

# train, test
# FASHION数据集提供了6万张28*28像素点的衣裤等图片和标签，用于训练
# FASHION数据集提供了1万张28*28像素点的衣裤等图片和标签，用于测试
fashion = tf.keras.datasets.fashion_mnist
(x_train, y_train),(x_test, y_test) = fashion.load_data()
# 对输入网络的输入特征进行归一化，把输入特征的数值变小更适合神经网络吸收
x_train, x_test = x_train / 255.0, x_test / 255.0

# model = tf.keras.models.Sequential
# 作为输入特征，输入神经网络时，将数据拉伸为一维数组
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# model.compile
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

# model.fit
model.fit(x_train, y_train, batch_size = 32, epochs=5, validation_data=(x_test, y_test), validation_freq=1)

# model.summary
model.summary()
