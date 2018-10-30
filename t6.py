
from PIL import Image
import glob

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten
import tensorflow.keras as keras
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist, cifar10

import numpy as np

import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)


# batch大小，每处理128个样本进行一次梯度更新
batch_size = 100
# 类别数
num_classes = 40
# 迭代次数
epochs = 2


def read_files(pathname):
    dir = glob.glob(pathname)
    x = []
    y = []
    for filename in dir:
        img = Image.open(filename).convert('L')
        #img = img.point(lambda x: [1, 0][x > 150], '1')
        #img.show()#;exit()
        x.append(np.array(img))
        y.append(list(filename[9:13]))
    x = np.array(x)
    y = np.array(y)
    return x, y

# 读取文件
x_train, y_train = read_files('dede/img/*.png')
x_test, y_test = read_files('dede/tst/*.png')

# print(x_train.shape);exit()

x_train = x_train.reshape(x_train.shape[0], 24, 68, 1)
x_test = x_test.reshape(x_test.shape[0], 24, 68, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255.0
x_test /= 255.0

# 转为one_hot 数据
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# 构建模型
model = Sequential()
# 第一层为二维卷积层
# 32 为filters卷积核的数目，也为输出的维度
# kernel_size 卷积核的大小，3x3
# 激活函数选为relu
# 第一层必须包含输入数据规模input_shape这一参数，后续层不必包含
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(24, 68, 1)))
# 再加一层卷积，64个卷积核
model.add(Conv2D(64, (3, 3), activation='relu'))
# 加最大值池化
model.add(MaxPooling2D(pool_size=(2, 2)))
# 加Dropout，断开神经元比例为25%
model.add(Dropout(0.25))
# 加Flatten，数据一维化
model.add(Flatten())
# 加Dense，输出128维
model.add(Dense(128, activation='relu'))
# 再一次Dropout
model.add(Dropout(0.5))
# 最后一层为Softmax，输出为10个分类的概率
model.add(Dense(num_classes, activation='softmax'))

# 配置模型，损失函数采用交叉熵，优化采用Adadelta，将识别准确率作为模型评估
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# 训练模型，载入数据，verbose=1为输出进度条记录
# validation_data为验证集
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

# 开始评估模型效果
# verbose=0为不输出日志信息
score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print(model.predict_classes(x_test))
