from PIL import Image
import numpy as np
from mynn import single, reader
import glob
import random
import tensorflow as tf


# rd = reader.ReadQueue('images/*.jpg')

# for x in range(3):
#     key, data = rd.next_batch(2)
#     print(key)


# img = Image.open('dede/tt/1_5bd6a47845aed.png')
# img = img.convert('L')
# img = img.point(lambda x: [0, 1][x > 100], '1')
# # print(img.histogram())
# print(np.sum(np.array(img)))

# print('dede/tt/1_5bd6a47845/ggg'.split('/')[-2])
hh = np.array(
    [['gg', 'ww', 'ghw', 'wfsaf', 'asf', 'gwfs'], [1, 2, 3, 4, 5, 6]])
print(hh)
aaa = hh.transpose()
print(aaa)
