'''
softmax 单层神经网络 对应简单的数据且对精度要求不高的数据，是不错的选择
优点：训练快。结构简单，调参方便。
'''

import tensorflow as tf
import numpy as np
from PIL import Image


class Nn():
    def __init__(self, pixel, kind):  # pixel 多少个像素  kind 多少种分类
        self._x = tf.placeholder(tf.float32, [None, pixel])  # 输入数据
        # 权重 注：希望能得到10个784维向量  10个数字（多少类）
        self._W = tf.Variable(tf.zeros([pixel, kind]))
        self._b = tf.Variable(tf.zeros([kind]))  # 偏置    注：10个数字 （多少类）
        self._y = tf.nn.softmax(
            tf.matmul(self._x, self._W) + self._b)  # y是最终预测的值
        self._y_ = tf.placeholder("float", [None, kind])  # 正确值

    # 准备数据
    def ready(self, learning_rate=0.01):
        # 得出交叉裔
        self.cross_entropy = -tf.reduce_sum(self._y_ * tf.log(self._y + 1e-10))
        # 梯度下降法：TensorFlow会用你选择的优化算法来不断地修改变量以降低成本
        self.train_step = tf.train.GradientDescentOptimizer(
            learning_rate).minimize(self.cross_entropy)  # 以0.01的学习速率最小化交叉熵

        # 评估
        correct_prediction = tf.equal(
            tf.argmax(self._y, 1), tf.argmax(self._y_, 1))
        # 是把bool转换为 float ；  tf.reduce_mean 计算平均值
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        # 初始化变量 因为是要放到 c写的核心里 所有这里要初始化变量
        init = tf.initialize_all_variables()
        # 开始一个回话，和核心进行交互
        self.sess = tf.Session()
        self.sess.run(init)
    # 训练

    def run(self, xs, ys):
        _, self.loss_ = self.sess.run([self.train_step, self.cross_entropy], feed_dict={
            self._x: xs, self._y_: ys})
    # 打印数据

    def print_accuracy(self, step, tx, ty):
        accuracy_ = self.sess.run(self.accuracy, {self._x: tx, self._y_: ty})
        print('Step:', step, '| train cross_entropy: %.4f' %
              self.loss_, '| test accuracy: %.2f' % accuracy_)

    # 测试数据

    def test(self, file_model, tx, ty=None):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            try:
                saver.restore(sess, file_model)
            except Exception as e:
                print('test must have a model file!', "error:", e)
                exit()
            prediction = sess.run(tf.argmax(self._y, 1),
                                  feed_dict={self._x: tx})
            print(prediction, 'prediction number')
            if type(ty) != type(None):
                print(np.argmax(ty, 1), 'real number')

    def test_from_img_file(self, file_model, file_img, label=None):
        img = Image.open(file_img)
        x_data = np.array(img).flatten() / 255.0  # 转换为array 并且展平为向量
        self.test(file_model, [x_data])
        if type(label) != type(None):
            print([label], 'real number')

    # 保存模型

    def save(self, filename):
        saver = tf.train.Saver()
        saver.save(self.sess, filename)
        return True

    # 关闭sess

    def close(self):
        self.sess.close()
