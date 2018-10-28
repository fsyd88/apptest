
import numpy as np
# import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data

# import struct as st

from PIL import Image
# import matplotlib.pyplot as plt


from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('mnist/data/', one_hot=True, dtype=np.uint8)

imgs, _ = mnist.train.next_batch(10)

# i = 0
# for img in imgs: 
#     i += 1
#     Image.fromarray(img.reshape(28, 28)).save(("images/{0}.jpg".format(i)))

# img = Image.open('images/1.jpg').convert('L')
# print(np.hstack(np.array(img)))

# Image.fromarray(first_img.reshape(28, 28)).show()

# res = open('mnist/data/t10k-images.idx3-ubyte', 'rb')

# bt = res.read(16+784)

# aa = st.unpack_from('>784B', bt, 16)


# print(np.array(aa))

# img = img.reshape(28, 28)

# tmp = np.zeros([28, 28])+255

# print(img)

# img = np.dot(img, 255)

# abc = np.multiply(np.array(aa, np.float32), 1.0/255.0)

# print(abc)

# print(np.multiply(abc, 255).astype(np.uint8))

# # img = Image.fromarray(.reshape((28, 28)))


# # img.save('88.jpg')

# #im = Image.open('2.jpg')

# #img = np.asarray(im)

# # print(np.array(aa))

# # print(np.asarray(im))

# # plt.ion()

# #

# #x = np.linspace(-1, 1, 50)

# # img.show()

# # print(np.array(Image.open('1.jpg')).shape)

# # plt.imshow(img)
# # plt.show()

# # img.save('1.jpg')
