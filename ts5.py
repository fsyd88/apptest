import numpy as np

a = np.arange(784)  # ->(784,)
# a = np.arange(784).reshape(28, 28)  # ->(28, 28)

# a = a[np.newaxis, :, :, np.newaxis]  # ->(1, 28, 28, 1)

# #a=a.ravel()  # -> (784,)
# a=a.flatten() # ->(784,)

a.resize(28,28)

<<<<<<< HEAD
# # np.set_printoptions(threshold=np.inf)
# # print(im)

# # 把文件名入list
filenames = glob.glob('images/*.jpg')


def  test(batch_size=2):
    with tf.Session() as sess:
        # string_input_producer会产生一个文件名队列
        filename_queue = tf.train.string_input_producer(
            filenames, shuffle=True, num_epochs=None)
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
                #key = sess.run(key)
                abc=[]
                for x in range(batch_size):
                    abc.append(sess.run(key))
                yield abc
                #
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)

hh=test(13)
for x in range(200):
    print(hh.__next__())
    print('------')

# for  x in test():
#     print(x)
#     print('------');


# #with tf.Session() as sess:

# dataset = tf.data.Dataset.from_tensor_slices(
#     {
#         "a": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),                                       
#         "b": np.random.uniform(size=(5, 2))
#     }
# )
# iterator = dataset.make_one_shot_iterator()
# one_element = iterator.get_next()
# with tf.Session() as sess:
#     for i in range(2):
#         print(sess.run(one_element))   

# m1=np.array([])
# m2=None
# if not m2:
#     print('aaa')
# else:
#     print('no')
=======
print(a.shape)
>>>>>>> 6d826b9bf29bf7193bc25a4524f59dd7bde011c5
