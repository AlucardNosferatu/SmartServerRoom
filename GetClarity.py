import os
import cv2
import numpy as np
from CamMonitor import calcAndDrawHist


def get_clarity():
    path = "C:\\Users\\16413\\Downloads\\pic(1)"
    for e, i in enumerate(os.listdir(path)):
        if i.endswith(".PNG"):
            img = cv2.imread(os.path.join(path, i))
            hist_img, hist = calcAndDrawHist(img, [255, 0, 255], None)
            dark = np.sum(np.squeeze(hist)[:128])
            bright = np.sum(np.squeeze(hist)[128:])
            dbr = dark / bright
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            image_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()  # 图像模糊度
            res = img.shape[0] * img.shape[1]
            clarity = int(image_var) * 10000 / res
            if dbr < 2 and clarity > 13:
                print(i, " ", clarity, " ", dbr)
            else:
                print(i, "neg")


def blurify():
    path = "Samples"
    for e, i in enumerate(os.listdir(path)):
        for size in [5, 11, 21]:
            if i.endswith(".jpg"):
                img = cv2.imread(os.path.join(path, i), 1)
                blur = cv2.GaussianBlur(img, (size, size), 0)
                cv2.imwrite(os.path.join(path, i).replace(".jpg", "_blur" + str(size) + ".jpg"), blur)


get_clarity()
