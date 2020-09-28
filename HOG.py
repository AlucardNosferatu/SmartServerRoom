import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

# 定义HOG对象，采用默认参数，或者按照下面的格式自己设置
defaultHog = cv2.HOGDescriptor()
# 设置SVM分类器，用默认分类器
defaultHog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


def detect_and_draw(roi):
    (rects, weights) = defaultHog.detectMultiScale(roi, winStride=(4, 4), padding=(8, 8), scale=1.05)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(roi, (xA, yA), (xB, yB), (0, 255, 0), 2)
    return roi
