from PIL import Image
import numpy as np
from mynn import single, reader
import glob
import random
import tensorflow as tf

from tensorflow import keras

# 第一种读取 csv文件
'''
读取 csv格式数据（可以是文件名：标签 ，也可以是 ndarray:标签）
file data like 
dede/img\0003_5bd7b37339fff.png,0003
dede/img\0007_5bd7b372235d2.png,0007
dede/img\0017_5bd7b36eb238c.png,0017
dede/img\0020_5bd7b36dcdc9c.png,0020
dede/img\0020_5bd7b36e21613.png,0020
'''


def read_csv_content():
    with open('a.txt') as f:
        content = f.read().split("\n")[:-1]  # 去掉最后一行空字符
    valuequeue = tf.train.string_input_producer(content, shuffle=True)
    with tf.Session() as sess:
        tf.train.start_queue_runners()  # 必须设置 否则 dequeue 会一直阻塞
        value = valuequeue.dequeue()
        dir, labels = tf.decode_csv(records=value, record_defaults=[
            ["string"], [""]], field_delim=",")
        print(sess.run([dir, labels]))


# 第二种 读取csv：
def read_csv_file():
    filename_queue = tf.train.string_input_producer(['a.txt'])
    rd = tf.TextLineReader()
    key, value = rd.read(filename_queue)
    # 如果某一列为空，指定默认值，同时指定了默认列的类型
    record_defaults = [['H'], ['K']]
    row = tf.decode_csv(value, record_defaults=record_defaults)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        # 主线程，消费50个数据
        for _ in range(50):
            print(sess.run(row))
        # 主线程计算完成，停止所有采集数据的进程
        coord.request_stop()
        # 指定等待某个线程结束
        coord.join(threads)


# 写入 TFRecord 文件
def write_tfrecord_file(filename):
    # 创建对象 用于向记录文件写入记录
    writer = tf.python_io.TFRecordWriter('data.tfrecord')

    dir = glob.glob('dede/img/*.png')
    for x in dir:
        label = str.encode(x[9:13])  # 字符串需要转为 bytes
        img = Image.open(x)
        # 将图片转化为原生bytes
        img_raw = img.tobytes()
        # 将数据整理成 TFRecord 需要的数据结构
        # "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index]))  多种格式：BytesList,Int64List,FloatList
        example = tf.train.Example(features=tf.train.Features(feature={
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label]))}
        ))
        # 序列化
        serialized = example.SerializeToString()
        # 写入文件
        writer.write(serialized)


# 读取 TFrecord
def read_tfrecord_file(filename):
    # 创建文件队列,不限读取的数量
    filename_queue = tf.train.string_input_producer([filename], shuffle=False)
    # 创建一个文件读取器 从队列文件中读取数据
    reader = tf.TFRecordReader()
    # reader从 TFRecord 读取内容并保存到 serialized_example中
    _, serialized_example = reader.read(filename_queue)
    # 读取serialized_example的格式
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.string),
            'img_raw': tf.FixedLenFeature([], tf.string)
        }
    )
    # 解析从 serialized_example 读取到的内容
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [24, 68, 3])
    label = features['label']
    return img, label


rd = reader.TFReader()

#rd.save('dede/img/*.png', 'abc.tfrecord')


# '''
# 测试
# '''


images_batch, labels_batch = rd.input_data(['./abc.tfrecord'],1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 创建一个协调器，管理线程
    coord = tf.train.Coordinator()
    #启动QueueRunner, 此时文件名才开始进队
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # print('开始训练!\n')
    try:
        img, label = sess.run([images_batch, labels_batch])
        Image.fromarray(img[0]).show()
        print(img,label)
    except tf.errors.OutOfRangeError as identifier:
        print('OutOfRangeError')
    finally:
        coord.request_stop()

    # 终止线程

    coord.join(threads)
