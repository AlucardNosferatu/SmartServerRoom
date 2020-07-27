import cv2
import numpy as np
from sklearn.cluster import KMeans

delta = 50
ma = 40
max_var = 1
distance_th = 30
w_th = 30
h_th = 30


def get_boxes(vis):
    # Create MSER object

    mser = cv2.MSER_create(_delta=delta, _min_area=ma, _max_variation=max_var)

    # Convert to gray scale
    gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)

    # detect regions in gray scale image
    regions, boxes = mser.detectRegions(gray)
    y_list = []
    x_list = []
    w_list = []
    h_list = []
    for box in boxes:
        x, y, w, h = box
        if w > w_th or h > h_th:
            continue
        if vis.shape[0] / 5 < y < 4 * vis.shape[0] / 5:
            continue
        y_list.append(y)
        x_list.append(x)
        w_list.append(w)
        h_list.append(h)
        # vis = cv2.rectangle(
        #     vis,
        #     (x, y),
        #     (x + w, y + h),
        #     (255, 0, 0),
        #     2
        # )

    y = np.array(y_list).reshape(-1, 1)
    km = KMeans(n_clusters=2)
    km.fit(y)
    labels = km.labels_.tolist()
    centers = km.cluster_centers_.tolist()
    rest_y_0 = []
    rest_y_1 = []
    rest_y = [rest_y_0, rest_y_1]
    rest_x_0 = []
    rest_x_1 = []
    rest_x = [rest_x_0, rest_x_1]

    for i in range(len(labels)):
        distance = abs(y_list[i] - centers[labels[i]][0])
        if distance > distance_th:
            # vis = cv2.rectangle(
            #     vis,
            #     (x_list[i], y_list[i]),
            #     (x_list[i] + w_list[i], y_list[i] + h_list[i]),
            #     (255, 0, 0),
            #     2
            # )
            pass
        else:
            rest_y[labels[i]].append(y_list[i])
            rest_x[labels[i]].append(x_list[i])
            # vis = cv2.rectangle(
            #     vis,
            #     (x_list[i], y_list[i]),
            #     (x_list[i] + w_list[i], y_list[i] + h_list[i]),
            #     (0, 0, 255),
            #     2
            # )
    y1 = np.mean(np.array(rest_y[0]))
    x1_1 = np.min(np.array(rest_x[0]))
    x1_2 = np.max(np.array(rest_x[0]))
    y2 = np.mean(np.array(rest_y[1]))
    x2_1 = np.min(np.array(rest_x[1]))
    x2_2 = np.max(np.array(rest_x[1]))
    # cv2.imshow('img', cv2.resize(vis, (1024, 768)))
    # cv2.waitKey()

    return [[y1, x1_1, x1_2], [y2, x2_1, x2_2]]


def cut_timestamp(cut_box, vis):
    for coordinates in cut_box:
        y_mean, x1, x2 = coordinates
        if y_mean - 10 < 0:
            y_mean = 0
        else:
            y_mean -= 10
        if x1 - 10 < 0:
            x1 = 0
        else:
            x1 -= 10
        if x2 + 40 > vis.shape[1] - 1:
            x2 = vis.shape[1] - 1
        else:
            x2 += 40
        vis = cv2.rectangle(vis, (x1, int(y_mean)), (x2, int(y_mean) + 40), (255, 255, 255), -1)
    return vis


if __name__ == '__main__':
    get_boxes()
