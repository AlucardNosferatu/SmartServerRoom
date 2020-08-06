import cv2
import numpy as np
from sklearn.cluster import KMeans
from EdgesDetection import get_edges


def sort_directions(lines):
    horizontal = []
    vertical = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dx >= 5 >= dy:
            horizontal.append([line[0]])
        elif dy >= 5 >= dx:
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


if __name__ == "__main__":
    sample = cv2.VideoCapture()
    while sample.isOpened():
        ret, frame = sample.read()
        if frame is not None:
            # frame = cv2.imread("Samples/Edges.jpg")
            # frame = cv2.resize(img, (768, 1024))
            lines = get_edges(frame)
            h, v = sort_directions(lines)
            u, d = km_1d(h, 'h')
            l, r = km_1d(v, 'v')
            for line in u:
                x1, y1, x2, y2 = line[0]
                frame = cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            for line in d:
                x1, y1, x2, y2 = line[0]
                frame = cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for line in l:
                x1, y1, x2, y2 = line[0]
                frame = cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            for line in r:
                x1, y1, x2, y2 = line[0]
                frame = cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.imshow("line_detect_possible_demo", frame)
            cv2.waitKey(1)
