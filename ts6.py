# import mynn.reader as rd

# q=rd.ReadQueue('images/*.jpg')

# for x in range(20):
#     print(q.next_batch(3))
#     print('----')

import tensorflow as tf
from PIL import Image
import numpy as np
from mynn import reader

# writer = tf.python_io.TFRecordWriter("train.tfrecords")

# img = Image.open('images/1.jpg')

# img_raw = img.tobytes()

# example = tf.train.Example(features=tf.train.Features(feature={
#     "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[1])),
#     'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
# }))
# writer.write(example.SerializeToString())


# for serialized_example in tf.python_io.tf_record_iterator("train.tfrecords"):
#     example = tf.train.Example()
#     example.ParseFromString(serialized_example)

#     image = example.features.feature['image'].bytes_list.value
#     label = example.features.feature['label'].int64_list.value
#     # 可以做一些预处理之类的
#     print(image, label)

# BATCH_SIZE = 4
# x = np.random.sample((10, 2))

# # make a dataset from a numpy array
# dataset = tf.data.Dataset.from_tensor_slices(x).batch(BATCH_SIZE)

# iter = dataset.make_one_shot_iterator()
# el = iter.get_next()

# with tf.Session() as sess:
#     print(sess.run(el))

# dataset=tf.data.TFRecordDataset('train.tfrecords')
# iter = dataset.make_one_shot_iterator()
# el = iter.get_next()

# with tf.Session() as sess:
#     print(sess.run(el))


# def read_and_decode(filename):  # 读入dog_train.tfrecords
#     filename_queue = tf.train.string_input_producer([filename])  # 生成一个queue队列

#     reader = tf.TFRecordReader()
#     _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
#     features = tf.parse_single_example(serialized_example,
#                                        features={
#                                            'label': tf.FixedLenFeature([], tf.int64),
#                                            'img_raw': tf.FixedLenFeature([], tf.string),
#                                        })  # 将image数据和label取出来

#     img = tf.decode_raw(features['img_raw'], tf.uint8)
#     img = tf.reshape(img, [128, 128, 3])  # reshape为128*128的3通道图片
#     img = tf.cast(img, tf.float32) * (1. / 255) - 0.5  # 在流中抛出img张量
#     label = tf.cast(features['label'], tf.int32)  # 在流中抛出label张量
#     return img, label


# print(read_and_decode('train.tfrecords'))

rd = reader.Reader()
# rd.save_records('test.td', 'images/*.jpg')
rd.read_records('test.td')
