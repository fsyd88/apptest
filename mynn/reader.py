'''
tensorflow读取文件的方法
'''
from PIL import Image
import numpy as np
import tensorflow as tf
import glob
import random


class ReadQueue:
    __this_data = None

    def __init__(self, files):
        self.filenames = glob.glob(files)
        if not self.filenames:
            raise ValueError(files+' no files!')

    # 读取习一组数据
    def __gen_data(self, batch_size):
        with tf.Session() as sess:
            # string_input_producer会产生一个文件名队列
            filename_queue = tf.train.string_input_producer(
                self.filenames, shuffle=True)
            # reader从文件名队列中读数据。对应的方法是reader.read
            reader = tf.WholeFileReader()
            # 读取一个文件名数据  key 文件名 value 文件数据
            key, value = reader.read(filename_queue)
            # tf.train.string_input_producer定义了一个epoch变量，要对它进行初始化
            tf.local_variables_initializer().run()
            # 使用start_queue_runners之后，才会开始填充队列
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            try:
                while True:
                    # 获取图片数据并保存
                    keys = []
                    datas = []
                    for _ in range(batch_size):
                        keys.append(key)
                        datas.append(value)
                    yield keys, datas
            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
            finally:
                coord.request_stop()
            coord.join(threads)

    # 利用队列的形式读取文件
    def next_batch(self, batch_size=10):
        if not self.__this_data:
            self.__this_data = self.__gen_data(batch_size)
        return self.__this_data.__next__()

# 使用 TFRecords 和 dataset


class TFReader():
    def __init__(self, slice_label=[9, 13]):
        self.slice_label = slice_label

    def __get_files(self, pathname):
        # 保存读取到的的文件和标签
        image_list = []
        label_list = []
        in_filenames = glob.glob(pathname)
        if not in_filenames:
            raise('no files!')
        for file in in_filenames:
            img = Image.open(file)
            image_list.append(img.tobytes())
            tmp_lab = file[self.slice_label[0]:self.slice_label[1]]
            label_list.append(str.encode(tmp_lab))

        # 保存打乱后的文件和标签
        images = []
        labels = []

        # 打乱文件顺序 连续打乱两次
        indices = list(range(len(image_list)))
        random.shuffle(indices)
        for i in indices:
            images.append(image_list[i])
            labels.append(label_list[i])
        random.shuffle([images, labels])
        print('样本长度为:', len(images))
        return images, labels

        # 保存到 record文件

    def save(self, pathname, savename):
        writer = tf.python_io.TFRecordWriter(savename)  # 要生成的文件
        images, labels = self.__get_files(pathname)

        for i in range(len(labels)):
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[labels[i]])),
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[images[i]]))
            }))  # example对象对label和image数据进行封装

            writer.write(example.SerializeToString())  # 序列化为字符串

        writer.close()

    def read_and_decode(self, filename, num_epochs=None, callback=None):
        if not isinstance(filename, list):
            raise ValueError('read_and_decode argument `filename` must list')

        filename_deque = tf.train.string_input_producer(filename)
        # 创建一个文件读取器 从队列文件中读取数据
        reader = tf.TFRecordReader()
        # reader从 TFRecord 读取内容并保存到 serialized_example中
        _, serialized_example = reader.read(filename_deque)

        # 读取serialized_example的格式
        features = tf.parse_single_example(
            serialized_example,
            features={
                'label': tf.FixedLenFeature([], tf.string),
                'image': tf.FixedLenFeature([], tf.string)
            })

        if callback:
            image, label = callback(features)
        else:
            image = tf.decode_raw(features['image'], tf.uint8)
            label = features['label']
        
        image = tf.reshape(image, [24, 68, 3])
        
        label = tf.cast(features['label'], tf.string)

        return image, label

    def input_data(self, filename, batch_size=128, num_epochs=None, capacity=4096, min_after_dequeue=1024, num_threads=10):
        '''
        读取小批量batch_size数据

        args:
            filenames:TFRecord文件路径组成的list
            num_epochs:每个数据集文件迭代轮数
            batch_size:小批量数据大小
            capacity:内存队列元素最大个数
            min_after_dequeue：内存队列元素最小个数
            num_threads：线城数
        '''
        '''
        读取批量数据  这里设置batch_size，即一次从内存队列中随机读取batch_size张图片，这里设置内存队列最小元素个数为1024，最大元素个数为4096    
        shuffle_batch 函数会将数据顺序打乱
        bacth 函数不会将数据顺序打乱
        '''
        img, label = self.read_and_decode(filename, num_epochs)

        images_batch, labels_batch = tf.train.shuffle_batch(
            [img, label], batch_size=batch_size, capacity=capacity, min_after_dequeue=batch_size*5, num_threads=num_threads)

        return images_batch, labels_batch


class WordVec():
    __number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    __alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                  'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    __v_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
                    'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    __char_idx_mappings = {}
    __idx_char_mappings = {}

    '''
        字符串one-hot互转
        char_type 数据类型 => 数，小，大，数小，小大，数小大
        max_captcha 默认4个字符
    '''

    def __init__(self, char_type, max_captcha=4):

        self.max_captcha = max_captcha
        captcha_chars = {
            1: self.__number,
            2: self.__alphabet,
            3: self.__v_alphabet,
            4: self.__number + self.__alphabet,
            5: self.__alphabet+self.__v_alphabet,
            6: self.__number + self.__alphabet + self.__v_alphabet
        }[char_type]

        for idx, c in enumerate(list(captcha_chars)):
            self.__char_idx_mappings[c] = idx
            self.__idx_char_mappings[idx] = c

        self.char_set_len = len(captcha_chars)

    def __convert(self, captcha_chars):
        char_idx_mappings = {}
        idx_char_mappings = {}
        for idx, c in enumerate(list(captcha_chars)):
            char_idx_mappings[c] = idx
            idx_char_mappings[idx] = c
        return char_idx_mappings, idx_char_mappings

    # 验证码转化为向量
    def text2vec(self, text):
        text_len = len(text)
        if text_len > self.max_captcha:
            raise ValueError('验证码最长%d个字符' % self.max_captcha)

        vector = np.zeros(self.max_captcha*self.char_set_len)
        for i, c in enumerate(text):
            idx = i * self.char_set_len + self.__char_idx_mappings[c]
            vector[idx] = 1
        return vector

    # 向量转化为验证码
    def vec2text(self, vec):
        text = []
        vec[vec < 0.5] = 0
        char_pos = vec.nonzero()[0]
        for i, c in enumerate(char_pos):
            char_idx = c % self.char_set_len
            text.append(self.__idx_char_mappings[char_idx])
        return text
