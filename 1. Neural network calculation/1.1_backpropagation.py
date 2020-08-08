# 建立神经网络模型，求解loss = (w-6)^2取得最小值时的参数值w
# 具体采用的是梯度下降法
# mengld
# 2020-8-7

import tensorflow as tf

w = tf.Variable(tf.constant(10, dtype = tf.float32))
lr = 0.3
epoch = 40

for epoch in range(epoch):  # for epoch定义了顶层循环，表示对数据集循环epoch次，初始化w为5，循环40次迭代
    with tf.GradientTape() as tape: # with结构对grads起了梯度的计算过程
        loss = tf.square(w - 6)
    grads = tape.gradient(loss, w)  #gradient函数用来计算函数对指定参数的求导

    w.assign_sub(lr * grads)    # assign_sub用来梯度下降计算，即 w -= lr*grads
    
    print("After %s epoch, w is %f, loss is %f." % (epoch, w.numpy(), loss))
