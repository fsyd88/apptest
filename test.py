import tensorflow as tf
import numpy as np

# 调试打印 tf 数据

def show(data):
    global sess
    print(sess.run(data))

# 开启回话
sess = tf.Session()

# x_data = tf.random_uniform([50], 1, 10)  # 这是错误的
x_data = np.random.rand(100).astype(np.float32)  # 输入数据必须是明确的 ，不能使用 tf的数据

y_data = x_data*0.6+15  # 正确值

W = tf.Variable(tf.ones([1]))  # 这个必须是个变量，训练的数据之一。(变量形状要与 正确值 的0.6 的那个形状一样)；

b = tf.Variable(tf.zeros([1]))  # 这个必须是个变量，训练的数据之一。(变量形状要与 正确值 的15 的那个形状一样)；

y = W*x_data+b  # 公式

# reduce_mean 最小平均值，如果不用 square(平方) ，就会出现负数，就会一直不能求出最小平均数了，所以值不能为负数
loss = tf.reduce_mean(tf.square(y-y_data))
# 优化函数
step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)  # 使用简单的梯度下降函数 来优化

init = tf.initialize_all_variables()  # 初始化所有变量
sess.run(init)  #

for i in range(220):
    sess.run(step)  # 训练
    if i % 20 == 0:
        print(i, sess.run(loss), sess.run(W), sess.run(b))
