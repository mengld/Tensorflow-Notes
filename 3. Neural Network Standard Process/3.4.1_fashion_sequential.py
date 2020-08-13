# 六步法之sequential搭建神经网络
# 用FASHION数据集识别衣裤
# mengld
# 2020-8-13

# import
import tensorflow as tf

# train, test
fashion = tf.keras.datasets.fashion_mnist
(x_train, y_train),(x_test, y_test) = fashion.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# model = tf.keras.models.Sequential
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
model.fit(x_train, y_train, batch_size = 32, epoch=5, validation_data=(x_test, y_test), validation_freq=1)

# model.summary
model.summary()
