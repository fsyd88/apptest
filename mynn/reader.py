'''
tensorflow读取文件的方法
'''
from PIL import Image
import numpy as np
import tensorflow as tf
import glob


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


class Reader():
    def __init__(self):
        pass

    # 保存到 record文件
    def save_records(self, savename, pathname):
        files = glob.glob(pathname)
        writer = tf.python_io.TFRecordWriter(savename)  # 要生成的文件
        for file in files:
            img = Image.open(file)
            img_raw = img.tobytes()  # 将图片转化为二进制格式
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[str.encode(file)])),
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))  # example对象对label和image数据进行封装
            writer.write(example.SerializeToString())  # 序列化为字符串
        writer.close()

    def read_and_decode_tfrecord(self, filename):
        filename_deque = tf.train.string_input_producer(filename)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_deque)
        features = tf.parse_single_example(serialized_example, features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string)})
        label = tf.cast(features['label'], tf.int32)
        img = tf.decode_raw(features['image'], tf.uint8)
        img = tf.reshape(img, [28, 28, 3])
        img = tf.cast(img, tf.float32) / 255.0
        return label, img

    # 获取迭代器
    def get_datta_iter(self, filename):
        dataset = tf.data.TFRecordDataset(filename)
        iter = dataset.make_one_shot_iterator()
        return iter
