from PIL import Image
import numpy as np
from mynn import single
import glob
import random

MAX_CAPTCHA = 1

CHAR_SET_LEN = 10


# 获取图片


def get_img(filename):
    img = Image.open(filename).convert('L')
    return np.array(img).flatten()/255

# 文本转向量


def text2vec(text):
    text_len = len(text)
    if text_len > MAX_CAPTCHA:
        raise ValueError('验证码最长4个字符')

    vector = np.zeros(MAX_CAPTCHA*CHAR_SET_LEN)

    def char2pos(c):
        #print(c)
        if c == '_':
            k = 62
            return k
        k = ord(c)-48
        if k > 9:
            k = ord(c) - 55
            if k > 35:
                k = ord(c) - 61
                if k > 61:
                    raise ValueError('No Map')
        return k
    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + char2pos(c)
        try:
            vector[idx] = 1
        except:
            print('aaaaaaa:',text)
    return vector

# 向量转文本


def vec2text(vec):
    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        char_idx = c % CHAR_SET_LEN
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx < 36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx - 36 + ord('a')
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)


dir = glob.glob('dede/img/*.png')

tf_x=[]
tf_y=[]
random.shuffle(dir)
for filename in dir:
    tf_x.append(get_img(filename))
    tf_y.append(text2vec(filename[9:10]))

tf_x=np.array(tf_x)
tf_y=np.array(tf_y)

dir = glob.glob('dede/test/*.png')

tx=[]
ty=[]

for filename in dir:
    tx.append(get_img(filename))
    ty.append(text2vec(filename[10:11]))

tx=np.asarray(tx)
ty=np.asarray(ty)

#print(tf_y[:3])

nn = single.Nn(28*28, 10)

nn.ready()

for i in range(500):
    index=i*30
    nn.run(tf_x[index:index+30], tf_y[index:index+30])
    if i % 20 == 0:
        #print(tf_x[index:index+30],tf_y[index:index+30])
        nn.print_accuracy(i, tx, ty)
#nn.save('model/t7.ckpt')


#print(train)
