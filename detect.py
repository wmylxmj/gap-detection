# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 19:23:03 2019
@author: wmy
"""

import numpy as np
import matplotlib.pyplot as plt
import pylab
from PIL import Image
from PIL import ImageFilter
import cv2 as cv

pylab.rcParams['figure.figsize'] = (10.0, 10.0)

image = Image.open('./test.bmp')
result = image.copy()
result = np.array(result)
result = result.reshape((result.shape[0], result.shape[1], 1))
result = result.repeat(3, axis=2)

x = np.array(image)
x = x > 127
x = 255 * (x + 0)
x = np.array(x, dtype=np.uint8)

x = x.reshape((x.shape[0], x.shape[1], 1))
x = x.repeat(3, axis=2)

cimage = cv.cvtColor(x, cv.COLOR_BGR2GRAY)
circles = cv.HoughCircles(cimage, cv.HOUGH_GRADIENT, 1, 2500, param1=80, param2=10, minRadius=400, maxRadius=800)

cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
circles = np.uint16(np.around(circles)) #around对数据四舍五入，转化为整数
print(circles)

circle = circles[0, 0]
xc = x.copy()
cv.circle(xc, (circle[0], circle[1]), circle[2], (0, 0, 255), 2)  #画圆
cv.circle(xc, (circle[0], circle[1]), 2, (255, 0, 0), 2)  #标记中心

cv.imshow("circles", xc)
cv.imwrite("circle.bmp", xc)

t = np.zeros(x.shape)
#画圆，参数为图片，中心点坐标，半径，颜色，线条类型：填充
cv.circle(t, (circle[0], circle[1]), circle[2], (1, 1, 1), -1)
xt = (255 - x) * t
xt = np.array(xt, dtype=np.uint8)
cv.imshow("xt", xt)
cv.imwrite("transer.bmp", xt)

gray = cv.cvtColor(xt, cv.COLOR_BGR2GRAY)
contours, _ = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

for i in range(len(contours)):
    area = cv.contourArea(contours[i])
    if area > 12:
        center = np.mean(np.squeeze(contours[i]), axis=0)
        center = np.uint16(np.around(center))
        cv.rectangle(xt, (center[0]-10, center[1]-10), (center[0]+10, center[1]+10), (255, 0, 0), 2)
        cv.rectangle(result, (center[0]-10, center[1]-10), (center[0]+10, center[1]+10), (0, 0, 255), 2)
        pass
    pass
cv.imshow("rect", xt)
cv.imshow("result", result)
cv.imwrite("result.png", result)

cv.waitKey(0)
cv.destroyAllWindows()

