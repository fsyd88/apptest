# import tensorflow as tf
# import glob




# # dir = glob.glob('dede/test/*.png')

# # tx = []
# # ty = []

# # for filename in dir:
# #     tx.append(filename)
# #     ty.append(filename[10:11])


# # tf.train.string_input_producer(tx)

# # image_raw_data = tf.gfile.FastGFile('images/2.jpg','rb').read()

# # image = tf.image.decode_jpeg(image_raw_data) #图片解码

# # # abs=tf.accumulate_n(tf.as_dtype)

# # # ts=tf.constant([1,2,3,4,5],tf.float32,shape=[2,3])

# # # ts =tf.range(1,10)

# # #tf.transpose(image,[0,1])
# # # image=tf.reshape(image,[-1])
# # # print(image.dtype)
# # # image=tf.reshape(image,[-1,1])
# # print(sess.run(image))

# dir = glob.glob('images/*.jpg')

# with tf.Session() as sess:

#     filename = {"abc":"ggg","www":"gwhg"}

#     # 文件名队列：string_input_producer会产生一个文件名队列
#     # num_epochs：一个local variable，需要通过local_variables_initializer来初始化
#     filename_queue = tf.train.string_input_producer(
#         filename, shuffle=True, num_epochs=None)

#     # print(sess.run(filename_queue))

#     reader = tf.WholeFileReader()
#     key, value = reader.read(filename_queue)

#     # image_tensor = tf.image.decode_jpeg(value)

#     tf.local_variables_initializer().run()

#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess, coord=coord)

#     i = 0
#     try:
#         while i < 20:
#             i += 1
#             # 获取图片转为tensor后的数据
#             # print(sess.run(tf.reshape(image_tensor[:10],[-1])))
#             print(i,key.eval())
#     except tf.errors.OutOfRangeError:
#         print('Done training -- epoch limit reached')
#     finally:
#         coord.request_stop()

#     coord.join(threads)



def next_batch(self, batch_size, fake_data=False, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    # if fake_data:
    #   fake_image = [1] * 784
    #   if self.one_hot:
    #     fake_label = [1] + [0] * 9
    #   else:
    #     fake_label = 0
    #   return [fake_image for _ in xrange(batch_size)], [
    #       fake_label for _ in xrange(batch_size)
    #   ]
    start = self._index_in_epoch      #0  50   10000
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:     #第一次获取
      perm0 = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm0)
      self._images = self.images[perm0]    ######################## 乱序数据
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:    #10050   start=9980  =10030
      # Finished epoch
      self._epochs_completed += 1    # 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start     # 10000-10000 =0
      images_rest_part = self._images[start:self._num_examples]   #[10000:10000]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]        #再次乱序
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return numpy.concatenate(
          (images_rest_part, images_new_part), axis=0), numpy.concatenate(
              (labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size    # 第一次是 0
      end = self._index_in_epoch             # = 0+50
      return self._images[start:end], self._labels[start:end]      #[0:50]  [50:100] [100:150]

'''
mnist next_batch 详解

设：数据长度:10000   batch_size =50
第一次时：
    1.乱序数据 整个数据组
    2.返回[start:end] =>[0:50]
第二次时：
    1.返回数据[start:end]=>[50:100]
.....
当 start+batch_size  大于 len(data) 时  比如:start=9980  batch_size=50
    1.取出最后20条数据（10000-9980）
    2.再次乱序
    3.重置 start = 0 ;  end= 30
    4.再取后面 30条数据
    5.合并 20条和30条数据
    6.返回
再次回到第二步:
    1.返回 [start:end] =>[30:80] 
如此重复
 '''