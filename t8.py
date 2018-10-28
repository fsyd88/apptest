'''
新版本tf，改版的cnn 网络 .新版速度更快，更简洁
from github of morvan zhou 
'''
import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from PIL import Image
import glob
import time
import random

BATCH_SIZE = 50         # 每次训练数量
LR = 0.001              # learning rate

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

for filename in dir:
    tx.append(get_img(filename))
    ty.append(text2vec(filename[10:11]))

test_x=np.asarray(tx)
test_y=np.asarray(ty)



# they has been normalized to range (0,1)
# mnist = input_data.read_data_sets('./data', one_hot=True)
# test_x = mnist.test.images[:2000]
# test_y = mnist.test.labels[:2000]


tf_x = tf.placeholder(tf.float32, [None, 28*28]) / 255.  # 输入数据存放地方
# (batch, height, width, channel)  转化输入数据
image = tf.reshape(tf_x, [-1, 28, 28, 1])
tf_y = tf.placeholder(tf.int32, [None, 10])            # input y (预测出的数据存放地方)

# CNN
conv1 = tf.layers.conv2d(   # shape (28, 28, 1)
    inputs=image,
    filters=16,
    kernel_size=5,
    strides=1,
    padding='same',
    activation=tf.nn.relu
)           # -> (28, 28, 16)

pool1 = tf.layers.max_pooling2d(
    conv1,
    pool_size=2,
    strides=2,
)           # -> (14, 14, 16)

conv2 = tf.layers.conv2d(pool1, 32, 5, 1, 'same',
                         activation=tf.nn.relu)    # -> (14, 14, 32)
pool2 = tf.layers.max_pooling2d(conv2, 2, 2)    # -> (7, 7, 32)

flat = tf.reshape(pool2, [-1, 7*7*32])          # -> (7*7*32, )
output = tf.layers.dense(flat, 10)              # output layer


h_fc1=tf.nn.relu(tf.matmul(output,W_fc1)+b_fc1)
h_fc1_drop=tf.nn.dropout(h_fc1,0.5)

loss = tf.losses.softmax_cross_entropy(
    onehot_labels=tf_y, logits=output)           # compute cost
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

# 评估函数
accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
    labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1),)[1]

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(
), tf.local_variables_initializer())  # the local var is for accuracy_op
sess.run(init_op)     # initialize var in graph

# following function (plot_with_labels) is for visualization, can be ignored if not interested


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

for step in range(300):
    b_x, b_y = next_batch(50)
    #print(step,b_x,b_y);exit()
    _, loss_ = sess.run([train_op, loss], {tf_x: b_x, tf_y: b_y})
    if step % 20 == 0:
        accuracy_, flat_representation = sess.run(
            [accuracy, flat], {tf_x: test_x, tf_y: test_y})
        print('Step:', step, '| train loss: %.4f' %
              loss_, '| test accuracy: %.2f' % accuracy_)

# print 10 predictions from test data
test_output = sess.run(output, {tf_x: test_x[:10]})
pred_y = np.argmax(test_output, 1)
print(pred_y, 'prediction number')
print(np.argmax(test_y[:10], 1), 'real number')
