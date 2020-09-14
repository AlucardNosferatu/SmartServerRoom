import cv2
import numpy as np

# 使用霍夫直线变换做直线检测，前提条件：边缘检测已经完成

__author__ = "boboa"

# 标准霍夫线变换
from sklearn.cluster import KMeans


def line_detection_demo(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)  # 函数将通过步长为1的半径和步长为π/180的角来搜索所有可能的直线
    for line in lines:
        rho, theta = line[0]  # line[0]存储的是点到直线的极径和极角，其中极角是弧度表示的
        a = np.cos(theta)  # theta是弧度
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))  # 直线起点横坐标
        y1 = int(y0 + 1000 * (a))  # 直线起点纵坐标
        x2 = int(x0 - 1000 * (-b))  # 直线终点横坐标
        y2 = int(y0 - 1000 * (a))  # 直线终点纵坐标
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow("image_lines", image)


# 统计概率霍夫线变换
def line_detect_possible_demo(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # 函数将通过步长为1的半径和步长为π/180的角来搜索所有可能的直线
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)
    k_list = []
    mid_points = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        xc = (x2 + x1) / 2
        yc = (y2 + y1) / 2
        if dx == 0:
            k = np.pi / 2
        else:
            k = np.arctan(dy / dx)
        k_list.append(k)
        mid_points.append([xc, yc])
    # mp_array = np.array(mid_points).reshape((-1, 2))
    k_array = np.array(k_list).reshape((-1, 1))
    # mp_and_k = np.hstack([mp_array, k_array])
    km = KMeans(n_clusters=2)
    km.fit(k_array)
    labels1 = km.labels_.tolist()

    direct1 = []
    d_indices1 = []
    direct2 = []
    d_indices2 = []
    for i in range(len(labels1)):
        cp = mid_points[i]
        if labels1[i] == 0:
            direct1.append(cp)
            d_indices1.append(i)
        elif labels1[i] == 1:
            direct2.append(cp)
            d_indices2.append(i)
    d_indices = [d_indices1, d_indices2]
    km = KMeans(n_clusters=2)
    km.fit(np.array(direct1).reshape(-1, 2))
    cc1 = km.cluster_centers_.tolist()
    labels2 = km.labels_.tolist()

    km = KMeans(n_clusters=2)
    km.fit(np.array(direct2).reshape(-1, 2))
    cc2 = km.cluster_centers_.tolist()
    labels3 = km.labels_.tolist()

    ccs = [cc1, cc2]
    labels = [labels2, labels3]
    for j in range(2):
        for k in range(2):
            image = cv2.circle(image, (int(ccs[j][k][0]), int(ccs[j][k][1])), 2, (0, 255, 255), 2)
        for i in range(len(labels[j])):
            x1, y1, x2, y2 = lines[d_indices[j][i]][0]
            if labels[j][i] == 0 and j == 0:
                image = cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            elif labels[j][i] == 1 and j == 0:
                image = cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            elif labels[j][i] == 0 and j == 1:
                image = cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            elif labels[j][i] == 1 and j == 1:
                image = cv2.line(image, (x1, y1), (x2, y2), (255, 255, 0), 2)

    cv2.imshow("line_detect_possible_demo", image)


def get_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # 函数将通过步长为1的半径和步长为π/180的角来搜索所有可能的直线
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)
    return lines


if __name__ == "__main__":
    img = cv2.imread("../Samples/LCD.jpg")
    # img = cv2.resize(img, (768, 1024))
    cv2.namedWindow("input image", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("input image", img)
    line_detect_possible_demo(img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
