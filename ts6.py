import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import glob
import cv2

np.set_printoptions(threshold=np.inf)

dir = glob.glob('dede/tst/*.png')

for k,f in enumerate(dir):
    s_img = Image.open(f)
    img = s_img.convert('L')
    img = img.point(lambda x: [255, 0][x > 160], '1')
    pix = img.load()
    w, h = img.size
    ver_list = []
    for x in range(w):  # ->68
        black = 0
        for y in range(h):  # ->24
            if pix[x, y] == 255:
                black += 1
        ver_list.append(black)
    text_contours = []
    start = end = 0
    for i, x in enumerate(ver_list):
        if x > 0 and start == 0:
            start = i
        elif start > 0 and x == 0:
            end = i
            text_contours.append([start, end])
            start = end = 0
    draw = ImageDraw.Draw(s_img)
    for contour in text_contours:
        draw.rectangle((contour[0]-2, 0, contour[1]+2, 23), outline='#F00')
    plt.subplot(4,3,k+1)
    plt.imshow(s_img)
    
    if k > 10:
        break

plt.show()
