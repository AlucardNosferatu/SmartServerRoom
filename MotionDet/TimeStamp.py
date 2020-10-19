import cv2
import numpy as np
from sklearn.cluster import KMeans

delta = 50
ma = 40
max_var = 1
distance_th = 30
w_th = 30
h_th = 30
edge_fraction = 6
mser = cv2.MSER_create(_delta=delta, _min_area=ma, _max_variation=max_var)


def get_boxes(vis):
    # Convert to gray scale
    gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
    # detect regions in gray scale image
    regions, boxes = mser.detectRegions(gray)
    y_list = []
    x_list = []
    w_list = []
    h_list = []
    # vis = cv2.rectangle(
    #     vis,
    #     (0, 0),
    #     (vis.shape[1] - 1, int(vis.shape[0] / edge_fraction) - 1),
    #     (0, 255, 0),
    #     2
    # )
    # vis = cv2.rectangle(
    #     vis,
    #     (0, int((edge_fraction - 1) * vis.shape[0] / edge_fraction) - 1),
    #     (vis.shape[1] - 1, vis.shape[0] - 1),
    #     (0, 255, 0),
    #     2
    # )
    for box in boxes:
        x, y, w, h = box
        if w > w_th or h > h_th:
            continue
        if vis.shape[0] / edge_fraction < y < (edge_fraction - 1) * vis.shape[0] / edge_fraction:
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

    y_temp = y[np.where(km.labels_ == 0)]
    intervals, y1_fm = fractional_mode(y_temp)
    # for y_int in range(intervals.shape[0] - 1):
    #     vis = cv2.rectangle(
    #         vis,
    #         (0, int(intervals[y_int])),
    #         (vis.shape[1] - 1, int(intervals[y_int + 1])),
    #         (0, 255, 0),
    #         1
    #     )
    y_temp = y[np.where(km.labels_ == 1)]
    intervals, y2_fm = fractional_mode(y_temp)
    # for y_int in range(intervals.shape[0] - 1):
    #     vis = cv2.rectangle(
    #         vis,
    #         (0, int(intervals[y_int])),
    #         (vis.shape[1] - 1, int(intervals[y_int + 1])),
    #         (255, 0, 0),
    #         1
    #     )
    # vis = cv2.rectangle(
    #     vis,
    #     (0, int(y1_fm)),
    #     (vis.shape[1] - 1, int(y2_fm)),
    #     (0, 0, 255),
    #     1
    # )
    # cv2.imshow('img', cv2.resize(vis, (1024, 768)))
    # cv2.waitKey()
    y_center_fm = [y1_fm, y2_fm]
    for i in range(len(labels)):
        # distance = abs(y_list[i] - centers[labels[i]][0])
        distance = abs(y_list[i] - y_center_fm[labels[i]])
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
    if vis is None:
        return vis
    else:
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


def fractional_mode(y):
    y_max = np.max(y)
    y_min = np.min(y)
    intervals = np.linspace(y_min - 1, y_max + 1, num=10)
    count = [0] * (intervals.shape[0] - 1)
    for i in range(intervals.shape[0] - 1):
        for j in range(y.shape[0]):
            if intervals[i] < y[j] <= intervals[i + 1]:
                count[i] += 1
    max_index = count.index(max(count))
    y = np.mean(intervals[max_index:max_index + 1])
    return intervals, y


if __name__ == '__main__':
    get_boxes()
