import numpy as np
import cv2

img=cv2.imread('images/3.png')
img=cv2.cvtColor(img,6)

_,img=cv2.threshold(img,170,255,cv2.THRESH_BINARY)

cv2.findContours()

cv2.namedWindow('test')
cv2.imshow('test',img)

cv2.waitKey(0)

cv2.findContours()