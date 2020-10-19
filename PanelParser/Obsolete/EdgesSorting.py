import cv2
import numpy as np
from numpy.linalg import norm
from sklearn.cluster import KMeans
from Obsolete.EdgesDetection import get_edges


def sort_directions(lines):
    horizontal = []
    vertical = []
    if lines is None:
        return horizontal, vertical
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dx >= 30 >= dy:
            horizontal.append([line[0]])
        elif dy >= 30 >= dx:
            vertical.append([line[0]])
    return horizontal, vertical


def km_1d(lines, mode='h'):
    xc_list = []
    yc_list = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        xc = (x2 + x1) / 2
        yc = (y2 + y1) / 2
        xc_list.append(xc)
        yc_list.append(yc)
    if mode == 'h':
        c_list = yc_list
    elif mode == 'v':
        c_list = xc_list
    else:
        raise ValueError('mode must be "h" or "v".')
    c_array = np.array(c_list).reshape((-1, 1))
    if c_array.shape[0] < 2:
        return [], []
    km = KMeans(n_clusters=2)
    km.fit(c_array)
    centers = km.cluster_centers_.tolist()
    labels = km.labels_.tolist()
    up_or_left = []
    down_or_right = []
    for i in range(len(labels)):
        if centers[labels[i]] > centers[1 - labels[i]]:
            down_or_right.append(lines[i])
        else:
            up_or_left.append(lines[i])
    return up_or_left, down_or_right


def get_4edges(img):
    lines = get_edges(img)
    h, v = sort_directions(lines)
    u, d = km_1d(h, 'h')
    l, r = km_1d(v, 'v')
    return u, d, l, r


def test_cam():
    sample = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    while sample.isOpened():
        ret, frame = sample.read()
        if frame is not None:
            # frame = cv2.imread("Samples/HMI2.jpg")
            # frame = cv2.resize(frame, (768, 1024))
            lines = get_edges(frame)
            h, v = sort_directions(lines)
            u, d = km_1d(h, 'h')
            l, r = km_1d(v, 'v')
            for line in u:
                x1, y1, x2, y2 = line[0]
                if ((x2 - x1) ** 2) + ((y2 - y1) ** 2) > (frame.shape[1] / 6) ** 2:
                    frame = cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            for line in d:
                x1, y1, x2, y2 = line[0]
                if ((x2 - x1) ** 2) + ((y2 - y1) ** 2) > (frame.shape[1] / 6) ** 2:
                    frame = cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for line in l:
                x1, y1, x2, y2 = line[0]
                if ((x2 - x1) ** 2) + ((y2 - y1) ** 2) > (frame.shape[0] / 6) ** 2:
                    frame = cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            for line in r:
                x1, y1, x2, y2 = line[0]
                if ((x2 - x1) ** 2) + ((y2 - y1) ** 2) > (frame.shape[0] / 6) ** 2:
                    frame = cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.imshow("line_detect_possible_demo", frame)
            cv2.waitKey(1)


if __name__ == "__main__":
    test_cam()
