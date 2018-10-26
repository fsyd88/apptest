import tensorflow as tf
import numpy as np

# 定义输入数据 x_data
x_data = np.random.rand(100)

# 定义正确标签 y_data
y_data = x_data * 0.3+5
# 定义权重和偏置 变量 W,b 使用tf.Variable()
W = tf.Variable(tf.zeros([1])+0.1)
b = tf.Variable(tf.zeros([1]))
# 损失函数 本例使用 最小平均数来做损失函数
y_ = x_data * W + b
loss = tf.reduce_mean(tf.square(y_-y_data))
# 梯度下降 本例使用tf.train.GradientDescentOptimizer(0.5)
cross=tf.train.GradientDescentOptimizer(0.5).minimize(loss)
# 初始化变量 到这步基本Picture已经完成
sess=tf.Session()
sess.run(tf.initialize_all_variables())
# 训练
for i in range(200):
    sess.run(cross)
    print(sess.run(W),sess.run(b))
