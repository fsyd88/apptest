'''
tensorflow读取文件的方法
'''
from PIL import Image
import numpy as np
import tensorflow as tf
import glob


class ReadQueue:
    def __init__(self, files):
        self.filenames = glob.glob(files)
        if not self.filenames:
            raise ValueError(files+' no files!')

    #读取习一组数据

    # 利用队列的形式读取文件
    def read_queue(dir_path, shuffle=True, num_epochs=5):
        filenames = glob.glob(dir_path)
        with tf.Session() as sess:
            # string_input_producer会产生一个文件名队列
            filename_queue = tf.train.string_input_producer(
                filenames, shuffle=shuffle, num_epochs=5)
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
                    image_data = sess.run(value)
                    print(image_data)
            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
            finally:
                coord.request_stop()
            coord.join(threads)
