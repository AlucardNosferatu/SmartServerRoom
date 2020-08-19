import cv2
import numpy as np
from EdgesFilter import get_u_d_l_r, get_k_and_b_and_c


def get_cross_point(line_a, line_b):
    _, _, k1, b1 = get_k_and_b_and_c(line_a)
    _, _, k2, b2 = get_k_and_b_and_c(line_b)
    x = (b2 - b1) / (k1 - k2)
    y = k1 * x + b1
    return [x, y]


def get_box(img):
    try:
        u_d_l_r = get_u_d_l_r(img)
        up, down, left, right = u_d_l_r
        ul = get_cross_point(up, left)
        ur = get_cross_point(up, right)
        dl = get_cross_point(down, left)
        dr = get_cross_point(down, right)
    except ValueError:
        ul = [0, 0]
        ur = [img.shape[1], 0]
        dr = [img.shape[1], img.shape[0]]
        dl = [0, img.shape[0]]
        return None
    points = np.array([ul, ur, dr, dl], np.int32)
    points = points.reshape((-1, 1, 2))
    return points


if __name__ == "__main__":
    sample = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    while sample.isOpened():
        ret, img = sample.read()
        if img is not None:
            # img = cv2.imread("Samples/HMI.jpg")
            box = get_box(img)
            if box is not None:
                img = cv2.polylines(img, [box], True, (0, 0, 255), 2)
                cv2.imshow("line_detect_possible_demo", img)
            else:
                cv2.imshow("line_detect_possible_demo", img)
            cv2.waitKey(1)
