import cv2
import numpy as np


def force_sharpen(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 定义一个核
    img = cv2.filter2D(img, -1, kernel=kernel)
    return img


if __name__ == "__main__":
    path = "Samples/watch7.jpg"
    img = cv2.imread(path)
    img = force_sharpen(img)
    cv2.imshow('fuck', img)
    cv2.waitKey()
