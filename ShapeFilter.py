import cv2
import numpy as np


def valid_shape(frame):
    frame = cv2.resize(frame, (512, 384))
    ret, thresh1 = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
    pixels = np.array(np.where(thresh1 == 255))
    # print(pixels.shape[1])
    # cv2.imshow('shape', thresh1)
    # cv2.waitKey()
    # print(pixels.shape[1])
    if 0 < pixels.shape[1] < 10:
        return True
    else:
        return False
