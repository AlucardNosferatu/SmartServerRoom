import cv2
import numpy as np
from math import sqrt
from Obsolete.EdgesSorting import get_4edges


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
    intervals = np.linspace(y_min - 1, y_max + 1, num=7)
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


def process_intervals(count, lines, mode):
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
    output_lines = process_intervals(count=count, lines=lines, mode=mode)
    return output_lines


def get_k_and_b_and_c(line):
    x1, y1, x2, y2 = line[0]
    xc = (x1 + x2) / 2
    yc = (y1 + y2) / 2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        dx = 1e-7
    k = dy / dx
    b = y2 - (k * x2)
    return xc, yc, k, b


def csl_with_slope(lines, img_h, img_w, mode='h'):
    xc_list = []
    yc_list = []
    slope_list = []
    bias_list = []
    for line in lines:
        xc, yc, k, b = get_k_and_b_and_c(line)
        slope_list.append(k)
        bias_list.append(b)
        xc_list.append(xc)
        yc_list.append(yc)
    elongated_endpoints_pos = []
    if mode == 'h':
        pivot = img_w / 2
    elif mode == 'v':
        pivot = img_h / 2
    else:
        raise ValueError("mode must be 'h' or 'v'.")
    for i in range(len(slope_list)):
        if mode == 'h':
            pos = slope_list[i] * pivot + bias_list[i]
        elif mode == 'v':
            pos = (pivot - bias_list[i]) / slope_list[i]
        else:
            raise ValueError("mode must be 'h' or 'v'.")
        elongated_endpoints_pos.append(pos)
    if len(elongated_endpoints_pos) == 0:
        return []
    count, intervals = interval_aggregation(y=np.array(elongated_endpoints_pos))
    output_lines = process_intervals(count=count, lines=lines, mode=mode)
    return output_lines


def out_most_only(up, down, left, right):
    u_d_l_r = []
    lines = [up, down, left, right]
    for i in range(4):
        temp = []
        for line in lines[i]:
            x1, y1, x2, y2 = line[0]
            temp2 = [-y1, y2, -x1, x2]
            score = temp2[i]
            temp.append(score)
        if len(temp) == 0:
            continue
        else:
            max_index = temp.index(max(temp))
            u_d_l_r.append(lines[i][max_index])
    return u_d_l_r


def get_u_d_l_r(img):
    up, down, left, right = get_4edges(img)
    # up = csl_via_centers(lines=up, mode='h')
    up = csl_with_slope(lines=up, img_h=img.shape[0], img_w=img.shape[1], mode='h')
    # up = length_filter(lines=up, img_h=img.shape[0], img_w=img.shape[1], mode='h')

    # down = csl_via_centers(lines=down, mode='h')
    down = csl_with_slope(lines=down, img_h=img.shape[0], img_w=img.shape[1], mode='h')
    # down = length_filter(lines=down, img_h=img.shape[0], img_w=img.shape[1], mode='h')

    # left = csl_via_centers(lines=left, mode='v')
    left = csl_with_slope(lines=left, img_h=img.shape[0], img_w=img.shape[1], mode='v')
    # left = length_filter(lines=left, img_h=img.shape[0], img_w=img.shape[1], mode='v')

    # right = csl_via_centers(lines=right, mode='v')
    right = csl_with_slope(lines=right, img_h=img.shape[0], img_w=img.shape[1], mode='v')
    # right = length_filter(lines=right, img_h=img.shape[0], img_w=img.shape[1], mode='v')

    u_d_l_r = out_most_only(up, down, left, right)
    return u_d_l_r


def test_cam():
    sample = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    while sample.isOpened():
        ret, img = sample.read()
        if img is not None:
            # img = cv2.imread("Samples/HMI2.jpg")
            up, down, left, right = get_4edges(img)
            # up = csl_via_centers(lines=up, mode='h')
            up = csl_with_slope(lines=up, img_h=img.shape[0], img_w=img.shape[1], mode='h')
            # up = length_filter(lines=up, img_h=img.shape[0], img_w=img.shape[1], mode='h')

            # down = csl_via_centers(lines=down, mode='h')
            down = csl_with_slope(lines=down, img_h=img.shape[0], img_w=img.shape[1], mode='h')
            # down = length_filter(lines=down, img_h=img.shape[0], img_w=img.shape[1], mode='h')

            # left = csl_via_centers(lines=left, mode='v')
            left = csl_with_slope(lines=left, img_h=img.shape[0], img_w=img.shape[1], mode='v')
            # left = length_filter(lines=left, img_h=img.shape[0], img_w=img.shape[1], mode='v')

            # right = csl_via_centers(lines=right, mode='v')
            right = csl_with_slope(lines=right, img_h=img.shape[0], img_w=img.shape[1], mode='v')
            # right = length_filter(lines=right, img_h=img.shape[0], img_w=img.shape[1], mode='v')

            u_d_l_r = out_most_only(up, down, left, right)
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
            for i in range(len(u_d_l_r)):
                x1, y1, x2, y2 = u_d_l_r[i][0]
                img = cv2.line(img, (x1, y1), (x2, y2), colors[i], 2)
            cv2.imshow("line_detect_possible_demo", img)
            cv2.waitKey()
            cv2.waitKey(1)


if __name__ == "__main__":
    test_cam()
