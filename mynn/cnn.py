import tensorflow as tf
import numpy as np
from PIL import Image


class Cnn():
    __x = None
    __y_ = None
    __img_info = None  # 图片信息 [w,h,channel]
    __kind = None

    # img_info 图片[w,h,channel]  kind 预测种类
    def __init__(self, img_info, kind=10, learning_rate=1e-3):
        self.__img_info = img_info
        self.__kind = kind
        self.__lenrning = learning_rate
        self.__x = tf.placeholder(
            tf.float32, [None, img_info[0]*img_info[1]])  # 输入数据
        self.__y_ = tf.placeholder(tf.float32, [None, self.__kind])

    # 权重

    def __weight_variable(self, shape):
        # 随机一个 0-0.1自己的值 到向量初始值
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    # 偏置初始化函数
    def __bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)  # 创建一个 值为 0.1的向量
        return tf.Variable(initial)

    # 2d卷积
    def __conv2d(self, x, W):
        # 向四面的步长 strides
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    # 2x2 池化
    def __max_pool_2x2(self, x):
        # 2x2池化模板
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 把图片转换为张量
    def get_x_image(self):
        # -1 重新计算张量大小，28 28 图片宽高，1颜色通道 (彩色rgb(为3))
        x_image = tf.reshape(
            self.__x, [-1, self.__img_info[0], self.__img_info[1], self.__img_info[2]])
        return x_image

    # 创建层  x_data 输入张量(第一次是图片 get_x_image()，第二层是第一次返回的池化数据)  ch_in  输入通道, ch_out 输出通道
    def conv_pool(self, x_data, ch_in, ch_out):
        W_conv = self.__weight_variable([5, 5, ch_in, ch_out])
        b_conv = self.__bias_variable([ch_out])  # 每一个输出通道 都对应一个偏置量
        # 把x_data和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max pooling 池化
        h_conv = tf.nn.relu(self.__conv2d(x_data, W_conv) + b_conv)
        h_pool = self.__max_pool_2x2(h_conv)
        return h_pool

    # ch_in 最后图片长*宽*高 ch_out 输出通道
    def full_conn(self, h_pool, ch_in=7*7*64, ch_out=1024):
        W_fc1 = self.__weight_variable([ch_in, ch_out])
        b_fc1 = self.__bias_variable([ch_out])  # 输出通道偏置

        h_pool2_flat = tf.reshape(h_pool, [-1, ch_in])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        self.__keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.__keep_prob)

        # 输出层
        W_fc2 = self.__weight_variable([ch_out, self.__kind])
        b_fc2 = self.__bias_variable([self.__kind])

        # 使用softmax 分类函数分类
        self.__y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    def ready(self):

        # 成本函数： 交叉裔 算法 公式：H(y)=Ey'log(y)
        self.__cross_entropy = -tf.reduce_sum(self.__y_*tf.log(self.__y_conv))

        correct_prediction = tf.equal(
            tf.argmax(self.__y_conv, 1), tf.argmax(self.__y_, 1))
        # bool convert to float, 求出最小平均值  评估函数
        self.__accuracy = tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32))

        # 用更加复杂的ADAM优化器来做梯度最速下降
        self.__train_step = tf.train.AdamOptimizer(
            self.__lenrning).minimize(self.__cross_entropy)
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

    def run(self, x, y, keep_drop=0.5):  # 训练
        _, self.__cross_entropy_r = self.sess.run([self.__train_step, self.__cross_entropy], feed_dict={
            self.__x: x, self.__y_: y, self.__keep_prob: keep_drop})

    def print_accuracy(self, step, tx, ty):
        accuracy_ = self.sess.run(
            self.__accuracy, {self.__x: tx, self.__y_: ty, self.__keep_prob: 1.0})
        print('Step:', step, '| train cross_entropy: %.4f' %
              self.__cross_entropy_r, '| test accuracy: %.2f' % accuracy_)

    # 测试数据
    def test(self, file_model, tx, ty=None):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            try:
                saver.restore(sess, file_model)
            except Exception as e:
                print('test must have a model file!', "error:", e)
                exit()
            prediction = sess.run(tf.argmax(self.__y_conv, 1),
                                  feed_dict={self.__x: tx, self.__keep_prob: 1.0})
            print(prediction, 'prediction number')
            if(type(ty) != type(None)):
                print(np.argmax(ty, 1), 'real number')

    # 测试图片数据
    def test_from_img_file(self, file_model, file_img, label=None):
        img = Image.open(file_img).convert('L')
        x_data = np.array(img).flatten() / 255.0  # 转换为array 并且展平为向量
        self.test(file_model, [x_data])
        if (type(label) != type(None)):
            print([label], 'real number')

    def save(self, filename):
        saver = tf.train.Saver()
        saver.save(self.sess, filename)
        return True

    def close(self):
        self.sess.close()
