import tensorflow as tf
import glob




# dir = glob.glob('dede/test/*.png')

# tx = []
# ty = []

# for filename in dir:
#     tx.append(filename)
#     ty.append(filename[10:11])


# tf.train.string_input_producer(tx)

# image_raw_data = tf.gfile.FastGFile('images/2.jpg','rb').read()

# image = tf.image.decode_jpeg(image_raw_data) #图片解码

# # abs=tf.accumulate_n(tf.as_dtype)

# # ts=tf.constant([1,2,3,4,5],tf.float32,shape=[2,3])

# # ts =tf.range(1,10)

# #tf.transpose(image,[0,1])
# # image=tf.reshape(image,[-1])
# # print(image.dtype)
# # image=tf.reshape(image,[-1,1])
# print(sess.run(image))

dir = glob.glob('images/*.jpg')

with tf.Session() as sess:

    filename = {"abc":"ggg","www":"gwhg"}

    # 文件名队列：string_input_producer会产生一个文件名队列
    # num_epochs：一个local variable，需要通过local_variables_initializer来初始化
    filename_queue = tf.train.string_input_producer(
        filename, shuffle=True, num_epochs=None)

    # print(sess.run(filename_queue))

    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)

    # image_tensor = tf.image.decode_jpeg(value)

    tf.local_variables_initializer().run()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    i = 0
    try:
        while i < 20:
            i += 1
            # 获取图片转为tensor后的数据
            # print(sess.run(tf.reshape(image_tensor[:10],[-1])))
            print(i,key.eval())
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
