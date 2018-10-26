'''
softmax 单层神经网络的 模型预测 测试
'''

import tensorflow as tf
import struct as st
import numpy as np

from PIL import Image

# 读取 mnist 文件的图片数据


def read_img():
    res = open('data/t10k-images.idx3-ubyte', 'rb')
    bt = res.read(16+784)
    img_arr = st.unpack_from('>784B', bt, 16)
    img_data = np.multiply(np.array(img_arr, np.float32), 1.0/255.0)
    return img_data

# 从文件读取 图片数据


def read_img_file(filename):
    img = Image.open(filename).convert('L')
    return np.multiply(np.hstack(np.array(img)), 1.0/255)


# 下面这三个参数 是必须的
x = tf.placeholder(tf.float32, [None, 784])  # 输入数据
W = tf.Variable(tf.zeros([784, 10]))  # 权重 注：希望能得到10个784维向量  10个数字（多少类）
b = tf.Variable(tf.zeros([10]))  # 偏置    注：10个数字 （多少类）
# 上面三个参数是服务 与下面这个 公式的
y = tf.nn.softmax(tf.matmul(x, W) + b)  # 回归函数  取最大概率 竟然 x,W 顺序不能调换

sess = tf.Session()

# 载入模型
saver = tf.train.Saver()
saver.restore(sess, 'model/smax.ckpt')

# 打印预测图片的 标签
prediction = sess.run(tf.argmax(y, 1), feed_dict={
                      x: [read_img_file('../images/3.jpg')]})
print(prediction)

sess.close()
