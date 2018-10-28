'''
tensorflow官网的cnn例子 使用的旧版tf
'''

#from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from PIL import Image
import glob
import time
import numpy as np
import random



def text2vec(text):
    vector = np.zeros(10)
    for i, c in enumerate(text):
        idx = i * 1 + (ord(c)-48)
        try:
            vector[idx] = 1
        except:
            print(text);exit()
            raise ValueError(text)
    return vector

def get_img(filename):
    img = Image.open(filename).convert('L')
    return np.array(img).flatten()/255


dir = glob.glob('dede/img/*.png')

file_data=[]
label_data=[]

label_count=0
random.shuffle(dir)
#print(dir);exit()
for filename in dir:
    file_data.append(filename)
    label_data.append(filename[9:10])
    label_count+=1
    # x_data.append(get_img(filename))
    # y_data.append(text2vec(filename[9:13]))

# x_data=np.array(x_data)
# y_data=np.array(y_data)

dir = glob.glob('dede/test/*.png')

tx=[]
ty=[]
random.shuffle(dir)
for filename in dir:
    tx.append(get_img(filename))
    ty.append(text2vec(filename[10:11]))

test_x=np.asarray(tx)
test_y=np.asarray(ty)


# 数据
#mnist = input_data.read_data_sets('data/', one_hot=True)
x = tf.placeholder(tf.float32, [None, 784])  # 输入数据

# 权重初始化函数


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  # 随机一个 0-0.1自己的值 到向量初始值
    return tf.Variable(initial)

# 偏置初始化函数


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)  # 创建一个 值为 0.1的向量
    return tf.Variable(initial)

# 卷积


def conv2d(x, W):
    # 向四面的步长 strides
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# 池化


def max_pool_2x2(x):
    # 2x2池化模板
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 测试 时读取文件
def read_img_file(filename):
    img = Image.open(filename).convert('L')
    return np.multiply(np.hstack(np.array(img)), 1.0/255)


# 第一层卷积和池化--------------
# 5*5 path 1输入通道数目 32输出通道数目(经验值(通过调整来达到最优识别))
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])  # 每一个输出通道 都对应一个偏置量

# -1 重新计算张量大小，28 28 图片宽高，1颜色通道 (彩色rgb(为3))
x_image = tf.reshape(x, [-1, 28, 28, 1])
# 把x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max pooling 池化
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二层-------------------
# 5*5path 32输入通道(上一层输出的通道) 输出64通道(经验值)
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])  # 每一个输出通道 都对应一个偏置量
# ReLu激活 并池化
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 密集连接层 -----------------
# 7*7的图是  2*2的池化层 两次池化后  64个输出通道， 1024个神经元
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])  # 输出通道偏置


h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

'''
为了减少过拟合，我们在输出层之前加入dropout。我们用一个placeholder来代表一个神经元的输出在dropout中保持不变的概率。
这样我们可以在训练过程中启用dropout，在测试过程中关闭dropout。 TensorFlow的tf.nn.dropout操作除了可以屏蔽神经元的输出外，
还会自动处理神经元输出值的scale。所以用dropout的时候可以不用考虑scale。
'''
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

# 使用softmax 分类函数分类
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 训练
y_ = tf.placeholder("float", [None, 10])
# 成本函数： 交叉裔 算法 公式：H(y)=Ey'log(y)
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
# 用更加复杂的ADAM优化器来做梯度最速下降  1e-4 = 0.0001
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.InteractiveSession()  # 启动session

# saver = tf.train.Saver()  # save and restore object
# saver.restore(sess, 'model/cnn.ckpt') #restore model

# 评估
# -> (y_conv == y_)->bool    y_conv 预测数据  y_ 一个占位符(训练时会传入)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
# bool convert to float, 求出最小平均值  评估函数
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())


index=0
def next_batch(size):
    #print(len(y_data),len(x_data));exit()
    global index
    end=index+size
    if end > label_count:
        end = label_count

    rx=[]
    ry=[]

    for x in range(index,end):
        rx.append(get_img(file_data[x]))
        ry.append(text2vec(label_data[x]))

    rx=np.array(rx)
    ry=np.array(ry)
    #print(index,end)
    index=end    
    return rx,ry


for i in range(300):
    batch = next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})  # 训练
    if i % 20 == 0:
        # 没100次训练，打印预测精准度
        train_accuracy = accuracy.eval(
            feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" %
              (i, train_accuracy))

# 最后打印测试数据的测试精准度   eval 和 sess.run() 意思一样
# print("test accuracy %g" % accuracy.eval(feed_dict={
#     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0
# }))


# saver.save(sess, 'model/cnn.ckpt')  #保存模型


# 测试真是图片数据
prediction = sess.run(tf.argmax(y_conv, 1), feed_dict={
                      x: [read_img_file('dede/test/2_85.png')], keep_prob: 1.0})
print(prediction)


sess.close()
