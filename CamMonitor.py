import cv2
import time
import numpy as np
import matplotlib.pyplot as plt


def enhance(f):
    # 线性变换
    a = 2
    o = float(a) * f
    o -= 50
    o[o > 255] = 255  # 大于255要截断为255
    o[o < 0] = 0
    # 数据类型的转换
    o = np.round(o)
    o = o.astype(np.uint8)
    return o


def calcAndDrawHist(image, color, mask=None):
    hist = cv2.calcHist([image], [0], mask, [256], [0.0, 255.0])
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)
    histImg = np.zeros([256, 256, 3], np.uint8)
    hpt = int(0.9 * 256)
    for h in range(256):
        intensity = int(hist[h] * hpt / maxVal)
        cv2.line(histImg, (h, 256), (h, 256 - intensity), color)
    return histImg, hist


def trigger(h_list, o_list, f_list, threshold, use_diff):
    signal = 0
    if np.array(o_list).any():
        s1_list = []
        s2_list = []
        for i in range(6):
            if use_diff:
                std1 = np.sum(h_list[i][64:, 0])
                std2 = np.sum(h_list[i][64:, 0])
            else:
                std1 = np.sqrt(np.sum(np.power(h_list[i] - o_list[i], 2)))
                std2 = np.sqrt(np.sum(np.power(h_list[i] - f_list[i], 2)))
            s1_list.append(std1)
            s2_list.append(std2)
        s3_list = np.multiply(np.array(s1_list), np.array(s2_list)).tolist()
        o_list = h_list
        signal = max(s3_list)
        print(signal)
        if signal < threshold:
            pos = -1
        else:
            pos = s3_list.index(signal)
    else:
        o_list = h_list
        pos = -1

    return pos, o_list, signal


old1 = None
old2 = None
old3 = None
old4 = None
old5 = None
old6 = None
first1 = None
first2 = None
first3 = None
first4 = None
first5 = None
first6 = None
record = []
old_frame = None
url = "http://admin:admin@10.80.84.47:8081"
# sample = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
# sample = cv2.VideoCapture("Sample.mp4")
sample = cv2.VideoCapture(url)
# plt.ion()  # 开启interactive mode 成功的关键函数
# plt.figure(1)
record = []
th_line = []
count = 0
th = 2e5
UseDiff = True
while sample.isOpened():
    sig = 0
    plt.clf()
    position = -1
    ret, frame = sample.read()
    th_line = [th] * 200
    if frame is not None:
        # frame = enhance(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if old_frame is not None and UseDiff:
            diff = frame.astype(np.int16) - old_frame.astype(np.int16)
            diff = np.abs(diff).astype(np.uint8)
            old_frame = frame.copy()
            frame = diff
        elif UseDiff:
            old_frame = frame.copy()
            diff = frame.astype(np.int16) - old_frame.astype(np.int16)
            diff = np.abs(diff).astype(np.uint8)
            frame = diff

        h = frame.shape[0]
        w = frame.shape[1]
        x1 = int(0.3 * w)
        x2 = int(0.7 * w)
        y = int(0.5 * h)

        # region get Histogram
        img_mask = np.zeros((h, w), np.uint8)
        img_mask[0:y, 0:x1] = 255
        hist_img1, hist1 = calcAndDrawHist(frame, (0, 0, 255), img_mask)
        # cv2.imshow("hist1", hist_img1)

        img_mask = np.zeros((h, w), np.uint8)
        img_mask[0:y, x1:x2] = 255
        hist_img2, hist2 = calcAndDrawHist(frame, (0, 0, 255), img_mask)
        # cv2.imshow("hist2", hist_img2)

        img_mask = np.zeros((h, w), np.uint8)
        img_mask[0:y, x2:w] = 255
        hist_img3, hist3 = calcAndDrawHist(frame, (0, 0, 255), img_mask)
        # cv2.imshow("hist3", hist_img3)

        img_mask = np.zeros((h, w), np.uint8)
        img_mask[y:h, 0:x1] = 255
        hist_img4, hist4 = calcAndDrawHist(frame, (0, 0, 255), img_mask)
        # cv2.imshow("hist1", hist_img1)

        img_mask = np.zeros((h, w), np.uint8)
        img_mask[y:h, x1:x2] = 255
        hist_img5, hist5 = calcAndDrawHist(frame, (0, 0, 255), img_mask)
        # cv2.imshow("hist2", hist_img2)

        img_mask = np.zeros((h, w), np.uint8)
        img_mask[y:h, x2:w] = 255
        hist_img6, hist6 = calcAndDrawHist(frame, (0, 0, 255), img_mask)
        # cv2.imshow("hist3", hist_img3)

        # endregion
        count += 1
        first_hist = first1 is None and first2 is None and first3 is None and first4 is None and first5 is None and first6 is None
        if first_hist or count > 200:
            count = 0
            first1 = hist1
            first2 = hist2
            first3 = hist3
            first4 = hist4
            first5 = hist5
            first6 = hist6

        hist_list = [hist1, hist2, hist3, hist4, hist5, hist6]
        old_list = [old1, old2, old3, old4, old5, old6]
        first_list = [first1, first2, first3, first4, first5, first6]
        p, old, sig = trigger(hist_list, old_list, first_list, th, UseDiff)
        old1, old2, old3, old4, old5, old6 = old

        if sig > 2 * th:
            sig = 2 * th
        record.append(sig)
        if len(record) > 200:
            del record[0]
        # plt.plot(record)
        # plt.plot(th_line)
        # plt.pause(0.005)
        position = p

        if position == 0:
            frame = cv2.rectangle(frame, (0, 0), (x1, y), (255, 255, 255), 5, cv2.LINE_AA)
        elif position == 1:
            frame = cv2.rectangle(frame, (x1, 0), (x2, y), (255, 255, 255), 5, cv2.LINE_AA)
        elif position == 2:
            frame = cv2.rectangle(frame, (x2, 0), (w, y), (255, 255, 255), 5, cv2.LINE_AA)
        elif position == 3:
            frame = cv2.rectangle(frame, (0, y), (x1, h), (255, 255, 255), 5, cv2.LINE_AA)
        elif position == 4:
            frame = cv2.rectangle(frame, (x1, y), (x2, h), (255, 255, 255), 5, cv2.LINE_AA)
        elif position == 5:
            frame = cv2.rectangle(frame, (x2, y), (w, h), (255, 255, 255), 5, cv2.LINE_AA)
        else:
            pass
        cv2.imshow("image", frame)
    else:
        break
    k = cv2.waitKey(50)
    # q键退出
    if k & 0xff == ord('q'):
        print("redefine threshold")
        th = th * 1.05
    if k & 0xff == ord('e'):
        print("redefine threshold")
        th = th * 0.95

sample.release()
cv2.destroyAllWindows()
