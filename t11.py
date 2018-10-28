import mynn.cnn
#from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from PIL import Image
import glob
import time
import random

# mnist = input_data.read_data_sets('data/', one_hot=True)


# nn = mynn.cnn.Cnn([28, 28, 1], 10)
# # 第一层
# cp1 = nn.conv_pool(nn.get_x_image(), 1, 32)
# # 第二层
# cp2 = nn.conv_pool(cp1, 32, 64)
# # 全连接层
# nn.full_conn(cp2)
# nn.ready()

# for i in range(200):
#     batch_xs, batch_ys = mnist.train.next_batch(50)
#     nn.run(batch_xs, batch_ys)
#     if i % 20 == 0:
#         nn.print_accuracy(i, mnist.test.images[:50], mnist.test.labels[:50])

# nn.save('model/cnn.ckpt')

# 测试
# nn.test('model/cnn.ckpt', mnist.test.images[:10], mnist.test.labels[:10])

# print(np.argmax(mnist.test.labels[:10],1))
# 从图片测试
# nn.test_from_img_file('model/cnn.ckpt', 'dede/222.png', 6)


def text2vec(text):
    vector = np.zeros(10)
    for i, c in enumerate(text):
        idx = i * 1 + (ord(c)-48)
        try:
            vector[idx] = 1
        except:
            print(text)
            exit()
            raise ValueError(text)
    return vector


def get_img(filename):
    img = Image.open(filename).convert('L')
    return np.array(img).flatten()/255


dir = glob.glob('dede/img/*.png')

file_data = []
label_data = []

label_count = 0
random.shuffle(dir)
for filename in dir:
    file_data.append(filename)
    label_data.append(filename[9:10])
    label_count += 1
    # x_data.append(get_img(filename))
    # y_data.append(text2vec(filename[9:13]))

# x_data=np.array(x_data)
# y_data=np.array(y_data)

dir = glob.glob('dede/test/*.png')

tx = []
ty = []

for filename in dir:
    tx.append(get_img(filename))
    ty.append(text2vec(filename[10:11]))

test_x = np.asarray(tx)
test_y = np.asarray(ty)


index = 0


def next_batch(size):
    # print(len(y_data),len(x_data));exit()
    global index
    end = index+size
    if end > label_count:
        end = label_count

    rx = []
    ry = []

    for x in range(index, end):
        rx.append(get_img(file_data[x]))
        ry.append(text2vec(label_data[x]))

    rx = np.array(rx)
    ry = np.array(ry)
    # print(index,end)
    index = end
    return rx, ry


nn = mynn.cnn.Cnn([28, 28, 1], 10)
# 第一层
cp1 = nn.conv_pool(nn.get_x_image(), 1, 32)
# 第二层
cp2 = nn.conv_pool(cp1, 32, 64)

# 全连接层
nn.full_conn(cp2)
# nn.ready()

# for i in range(500):
#     b_x, b_y = next_batch(50)
#     nn.run(b_x, b_y)
#     if i % 20 == 0:
#         nn.print_accuracy(i, test_x, test_y)
# nn.save('model/cnn2.ckpt')

nn.test('model/cnn2.ckpt', test_x[:50], test_y[:50])