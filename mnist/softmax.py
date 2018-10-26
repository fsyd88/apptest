'''
softmax 单层神经网络 对应简单的数据且对精度要求不高的数据，是不错的选择
优点：训练快。结构简单，调参方便。
'''

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


mnist = input_data.read_data_sets('data/', one_hot=True)

# 回归模型 公式:  y=softmax(Wx+b)  w：权重 x：输入数据  b：偏置


# 第一个数据类型(32位的浮点型)，第二个形状(图片数量,没个图片像素)  代码意思是：任意个784维向量(像素)的图片
x = tf.placeholder(tf.float32, [None, 784])  # 输入数据

W = tf.Variable(tf.zeros([784, 10]))  # 权重 注：希望能得到10个784维向量  10个数字（多少类）
b = tf.Variable(tf.zeros([10]))  # 偏置    注：10个数字 （多少类）
'''
注意:
W的维度是[784，10]，因为我们想要用784维的图片向量乘以它 以得到一个10维的"证据值向量"，每一位对应不同数字类。
b的形状是[10]，所以我们可以直接把它加到输出上面。
'''
y = tf.nn.softmax(tf.matmul(x, W) + b)  # y是最终预测的值

# tf.matmul(​​X，W)表示x乘以W，对应之前等式里面的

# 成本函数： 交叉裔 算法 公式：H(y)=Ey'log(y)
y_ = tf.placeholder("float", [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))  # 得出交叉裔
# 梯度下降法：TensorFlow会用你选择的优化算法来不断地修改变量以降低成本
train_step = tf.train.GradientDescentOptimizer(
    0.01).minimize(cross_entropy)  # 以0.01的学习速率最小化交叉熵

# 初始化变量 因为是要放到 c写的核心里 所有这里要初始化变量
init = tf.initialize_all_variables()
# 开始一个回话，和核心进行交互
sess = tf.Session()
sess.run(init)

# 然后开始训练模型，这里我们让模型循环训练1000次！  每次训练100个图片
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 保存模型
# saver = tf.train.Saver()
# saver.restore(sess, 'model/smax.ckpt')

# saver=tf.train.Saver()
# saver.save(sess,'model/smax.ckpt')

# 评估模型
# 使用tf.argmax函数 找出那些预测正确的标签
# 使用tf.equal 函数 来检测我们的预测

# tf.argmax(y,1)预测标签  tf.argmax(y_,1)正确标签  y_ 正确数据是使用占位符 最后传进去的
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# tf.cast(correct_prediction, "float") 是把bool转换为 float ；  tf.reduce_mean 计算平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# 打印测试数据测试的 预测正确的平均值
print(sess.run(accuracy, feed_dict={
      x: mnist.test.images, y_: mnist.test.labels}))
