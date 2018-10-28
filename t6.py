from tensorflow.examples.tutorials.mnist import input_data

from mynn import single

import numpy as np

mnist = input_data.read_data_sets('data/', one_hot=True)

nn = single.Nn(28*28, 10)
mnist.train.next_batch(50)
# 训练
# nn.ready()
# for i in range(2000):
#     batch_xs, batch_ys = mnist.train.next_batch(50)
#     nn.run(batch_xs, batch_ys)
#     if i % 100 == 0:
#         nn.print_accuracy(i, mnist.test.images[:50], mnist.test.labels[:50])
# nn.save('model/t6.ckpt')

# 测试
# nn.test('model/t6.ckpt',mnist.test.images[:50],mnist.test.labels[:50])
# 从图片测试
# nn.test_from_img_file('model/t6.ckpt', 'images/6.jpg', 6)
