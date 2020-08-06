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
        if 0.99 * th > distance > th / 3:
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


def csl_via_centers(lines, mode='h'):
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
        yx1_list = []
        yx2_list = []
        if len(count[i]) >= 1:
            for index in count[i]:
                x1, y1, x2, y2 = lines[index][0]
                if x1 > x2:
                    x2 += x1
                    x1 -= x2
                    x2 += x1
                    x1 = (-x1)
                if y1 > y2:
                    y2 += y1
                    y1 -= y2
                    y2 += y1
                    y1 = (-y1)
                if mode == "h":
                    xy1_list.append(x1)
                    xy2_list.append(x2)
                    yx1_list.append(y1)
                    yx2_list.append(y2)
                elif mode == "v":
                    xy1_list.append(y1)
                    xy2_list.append(y2)
                    yx1_list.append(x1)
                    yx2_list.append(x2)
                else:
                    raise ValueError("mode must be 'h' or 'v'.")
            min_xy1 = min(xy1_list)
            max_xy2 = max(xy2_list)
            min_yx1 = yx1_list[xy1_list.index(min_xy1)]
            max_yx2 = yx2_list[xy2_list.index(max_xy2)]
            if mode == "h":
                x1 = min_xy1
                x2 = max_xy2
                y1 = min_yx1
                y2 = max_yx2
                # y1 = int((intervals[i] + intervals[i + 1]) / 2)
                # y2 = y1
            elif mode == "v":
                y1 = min_xy1
                y2 = max_xy2
                x1 = min_yx1
                x2 = max_yx2
                # x1 = int((intervals[i] + intervals[i + 1]) / 2)
                # x2 = y1
            else:
                raise ValueError("mode must be 'h' or 'v'.")
            temp_line = [x1, y1, x2, y2]
            output_lines.append(np.array(temp_line).reshape(1, -1))
    return output_lines


def csl_with_slope(lines, mode='h'):

    pass


if __name__ == "__main__":
    img = cv2.imread("Samples/HMI2.jpg")
    up, down, left, right = get_4edges(img)

    up = csl_via_centers(lines=up, mode='h')
    # up = length_filter(lines=up, img_h=img.shape[0], img_w=img.shape[1], mode='h')

    down = csl_via_centers(lines=down, mode='h')
    down = length_filter(lines=down, img_h=img.shape[0], img_w=img.shape[1], mode='h')

    left = csl_via_centers(lines=left, mode='v')
    left = length_filter(lines=left, img_h=img.shape[0], img_w=img.shape[1], mode='v')

    right = csl_via_centers(lines=right, mode='v')
    right = length_filter(lines=right, img_h=img.shape[0], img_w=img.shape[1], mode='v')

    for line in up:
        x1, y1, x2, y2 = line[0]
        img = cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    for line in down:
        x1, y1, x2, y2 = line[0]
        img = cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for line in left:
        x1, y1, x2, y2 = line[0]
        img = cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    for line in right:
        x1, y1, x2, y2 = line[0]
        img = cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
    cv2.imshow("line_detect_possible_demo", img)
    cv2.waitKey()
