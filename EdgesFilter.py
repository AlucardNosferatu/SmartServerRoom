import cv2
import numpy as np
from math import sqrt
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


def interval_aggregation(y):
    y_max = np.max(y)
    y_min = np.min(y)
    intervals = np.linspace(y_min - 1, y_max + 1, num=10)
    count = []
    for i in range(intervals.shape[0] - 1):
        lb = intervals[i]
        up = intervals[i + 1]
        temp = []
        for j in range(y.shape[0]):
            if lb < y[j] <= up:
                temp.append(j)
        count.append(temp)
    return count, intervals


def combine_short_lines(lines, mode='h'):
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
    elif mode == "v":
        c_list = xc_list
    else:
        raise ValueError("mode must be 'h' or 'v'.")
    count, intervals = interval_aggregation(y=np.array(c_list))
    output_lines = []
    for i in range(len(count)):
        xy1_list = []
        xy2_list = []
        if len(count[i]) >= 1:
            for index in count[i]:
                x1, y1, x2, y2 = lines[index][0]
                if x1 > x2:
                    x2 += x1
                    x1 -= x2
                    x2 += x1
                    x1 = (-x1)
                if mode == "h":
                    xy1_list.append(x1)
                    xy2_list.append(x2)
                elif mode == "v":
                    xy1_list.append(y1)
                    xy2_list.append(y2)
                else:
                    raise ValueError("mode must be 'h' or 'v'.")
            min_xy1 = min(xy1_list)
            max_xy2 = max(xy2_list)
            if mode == "h":
                x1 = min_xy1
                x2 = max_xy2
                # y1 = int((intervals[i] + intervals[i + 1]) / 2)
                # y2 = y1
            elif mode == "v":
                y1 = min_xy1
                y2 = max_xy2
                # x1 = int((intervals[i] + intervals[i + 1]) / 2)
                # x2 = y1
            else:
                raise ValueError("mode must be 'h' or 'v'.")
            temp_line = [x1, y1, x2, y2]
            output_lines.append(np.array(temp_line).reshape(1, -1))
    return output_lines


if __name__ == "__main__":
    img = cv2.imread("Samples/LCD.jpg")
    u, d, l, r = get_4edges(img)
    d = combine_short_lines(lines=d, mode='h')
    # d = length_filter(lines=d, img_h=img.shape[0], img_w=img.shape[1], mode='h')
    for line in d:
        x1, y1, x2, y2 = line[0]
        img = cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow("line_detect_possible_demo", img)
    cv2.waitKey()
