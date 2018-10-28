from PIL import Image
import numpy as np
import tensorflow as tf
import glob


# # # dir = glob.glob('dede/tt/*.png')

# # # for f in dir:
# # #     img = Image.open(f).convert('L')
# # #     img = img.point(lambda x: [1, 0][x > 150], '1')
# # #     img.save(f.replace('dede/tt','dede/tt/2'))

# # img=Image.open('dede/tt/2/CWMX_5bd54bf6af6df.png').convert('L')
# # im=np.array(img)/255.0
# # #img.show()

# # np.set_printoptions(threshold=np.inf)
# # print(im)

# # 把文件名入list
# # filenames = glob.glob('dede/tt/2/*.png')

# with tf.Session() as sess:
#     # string_input_producer会产生一个文件名队列
#     filename_queue = tf.train.string_input_producer(
#         filenames, shuffle=False, num_epochs=5)
#     # reader从文件名队列中读数据。对应的方法是reader.read
#     reader = tf.WholeFileReader()
#     # 读取一个文件名数据  key 文件名 value 文件数据
#     key, value = reader.read(filename_queue)
#     # tf.train.string_input_producer定义了一个epoch变量，要对它进行初始化
#     tf.local_variables_initializer().run()
#     # 使用start_queue_runners之后，才会开始填充队列
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#     try:
#         while True:
#             # 获取图片数据并保存
#             image_data = sess.run(value)
#             print(image_data)
#     except tf.errors.OutOfRangeError:
#         print('Done training -- epoch limit reached')
#     finally:
#         coord.request_stop()
#     coord.join(threads)

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

m1=np.array([])
m2=None
if not m2:
    print('aaa')
else:
    print('no')