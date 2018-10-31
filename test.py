import tensorflow as tf
from mynn import reader, cnn
rd = reader.TFReader()


# rd.save('dede/img/*.png','data.td')
# rd.save('dede/tst/*.png','test.td')

x_batch, y_batch = rd.input_data(['data.td'])

x_test, y_test = rd.input_data(['test.td'], 200)

print(x_batch.shape, y_batch.shape, x_test.shape, y_test.shape)

# CNN
conv1 = tf.layers.conv2d(   # shape (28, 28, 1)
    inputs=x_batch,
    filters=16,
    kernel_size=5,
    strides=1,
    padding='same',
    activation=tf.nn.relu
)           # -> (28, 28, 16)

pool1 = tf.layers.max_pooling2d(
    conv1,
    pool_size=2,
    strides=2,
)           # -> (14, 14, 16)

conv2 = tf.layers.conv2d(pool1, 32, 5, 1, 'same',
                         activation=tf.nn.relu)    # -> (14, 14, 32)
pool2 = tf.layers.max_pooling2d(conv2, 2, 2)    # -> (7, 7, 32)

flat = tf.reshape(pool2, [-1, 7*7*32])          # -> (7*7*32, )
output = tf.layers.dense(flat, 10)              # output layer

loss = tf.losses.softmax_cross_entropy(
    onehot_labels=y_batch, logits=output)           # compute cost
train_op = tf.train.AdamOptimizer(0.01).minimize(loss)

# 评估函数
accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
    labels=tf.argmax(y_batch, axis=1), predictions=tf.argmax(output, axis=1),)[1]




sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(
), tf.local_variables_initializer())  # the local var is for accuracy_op
sess.run(init_op)     # initialize var in graph

# following function (plot_with_labels) is for visualization, can be ignored if not interested

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)

try:
    for step in range(100):
        _, loss_ = sess.run([train_op, loss])
        if step % 50 == 0:
            accuracy_, flat_representation = sess.run(
                [accuracy, flat])
            print('Step:', step, '| train loss: %.4f' %
                loss_, '| test accuracy: %.2f' % accuracy_)
except tf.errors.OutOfRangeError as identifier:
    print('aaaa')
finally:
    coord.request_stop()


coord.join(threads)

# print 10 predictions from test data
# test_output = sess.run(output)
# pred_y = np.argmax(test_output, 1)
# print(pred_y, 'prediction number')
# print(np.argmax(test_y[:10], 1), 'real number')
