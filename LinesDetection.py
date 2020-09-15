import cv2
import numpy as np
import math


def xy2length(line):
    x1, y1, x2, y2 = line[0]
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    length = math.sqrt(dx ** 2 + dy ** 2)
    return length


image = cv2.imread('Samples/Masked.jpeg')
gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)
cv2.imshow('edges', edges)
# 函数将通过步长为1的半径和步长为π/180的角来搜索所有可能的直线
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=5, maxLineGap=1)
lines = lines.tolist()
image_copy = image.copy()
for line in lines:
    x1, y1, x2, y2 = line[0]
    image_copy = cv2.line(image_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
cv2.imshow('lines', image_copy)
cv2.waitKey()
lengths = [xy2length(line) for line in lines]
for i in range(2):
    index = lengths.index(max(lengths))
    length = lengths.pop(index)
    x1, y1, x2, y2 = lines.pop(index)[0]
    image = cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    print(length)
cv2.imshow('lines', image)
cv2.waitKey()
