import os
import cv2
import numpy as np

file_path = "Samples"


# file_path = "Samples\\Sample (35).jpg"
def show_result(image_path):
    img = cv2.imread(image_path, 1)
    cv2.imshow("image", img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    upper = 132
    lower = 64
    while True:
        k = cv2.waitKey(50)
        mask = cv2.inRange(img, np.array([lower, 0, 0]), np.array([upper, 255, 255]))
        # img_res = cv.bitwise_and(img_med, img_med, mask=mask)
        # mask = cv2.Sobel(mask, cv2.CV_64F, 1, 1, ksize=9)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        if k & 0xff == ord('e'):
            mask = cv2.dilate(mask, kernel)
        if k & 0xff == ord('r'):
            mask = cv2.dilate(mask, kernel)
            mask = cv2.erode(mask, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=5)
        cv2.imshow("mask", cv2.resize(mask, (512, 384)))
        if k & 0xff == ord('w'):
            upper += 1
            if upper > 179:
                upper = 179
            print(upper)
        if k & 0xff == ord('s'):
            upper -= 1
            if upper < 0:
                upper = 0
            print(upper)
        if k & 0xff == ord('a'):
            lower += 1
            if lower > 179:
                lower = 179
            print(lower)
        if k & 0xff == ord('d'):
            lower -= 1
            if lower < 0:
                lower = 0
            print(lower)
        if k & 0xff == ord('q'):
            print("最终下限：", lower)
            print("最终上限：", upper)
            break


for e, i in enumerate(os.listdir(file_path)):
    if i.startswith("DustCap"):
        show_result(os.path.join(file_path, i))
