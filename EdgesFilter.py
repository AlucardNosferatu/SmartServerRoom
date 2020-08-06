from math import sqrt

import cv2
from EdgesSorting import get_4edges


def length_filter(lines, img_h, img_w, mode='h'):
    if mode == 'h':
        th = img_w
    elif mode == "v":
        th = img_h
    else:
        raise ValueError("mode must be 'h' or 'v'.")
    new_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        distance = sqrt((dx ** 2) + (dy ** 2))
        if 0.9 * th > distance > th / 5:
            new_lines.append(line)
    return new_lines


def combine_short_lines(lines, img_h, img_w, mode='h'):
    xc_list = []
    yc_list = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        xc = (x1 + x2) / 2
        yc = (y1 + y2) / 2
        xc_list.append(xc)
        yc_list.append(yc)
    if mode == "h":
        c_list = yc_list
        th = img_h
    elif mode == "v":
        c_list = xc_list
        th = img_w
    else:
        raise ValueError("mode must be 'h' or 'v'.")
    max_c = max(c_list)
    if max_c + 1 < th:
        max_c += 1
    min_c = min(c_list)
    if min_c - 1 > 0:
        min_c -= 1


if __name__ == "__main__":
    img = cv2.imread("Samples/LCD.jpg")
    u, d, l, r = get_4edges(img)
    # d = length_filter(lines=d, img_h=img.shape[0], img_w=img.shape[1], mode='h')
    for line in d:
        x1, y1, x2, y2 = line[0]
        img = cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow("line_detect_possible_demo", img)
    cv2.waitKey()
