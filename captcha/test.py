import tensorflow as tf

image_raw_data = tf.gfile.FastGFile('./1.jpg','rb').read()

print(image_raw_data)