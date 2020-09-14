import math

import cv2

img = cv2.imread("Samples/watch7.jpg", 3)
img_param = img.shape[0] * img.shape[1]
blur_param = int(img_param / 90000)
Canny_param_1 = int(img_param / 375)
Canny_param_2 = int(img_param / 1875)
cnt_length = int(img_param / 2500)
area_param_1 = int(img_param / 6)
area_param_2 = int(img_param / 1.5)
img = cv2.blur(img, (blur_param, blur_param))

img_gray = cv2.Canny(img, Canny_param_1, Canny_param_2)  # Canny边缘检测，参数可更改
ret, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("0", thresh)
cv2.waitKey()
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # contours为轮廓集，可以计算轮廓的长度、面积等
for cnt in contours:
    if len(cnt) > cnt_length:
        ell = cv2.fitEllipse(cnt)
        area = math.pi * ell[1][0] * ell[1][1] / 4
        if area_param_1 < area < area_param_2:
            img = cv2.ellipse(img, ell, (0, 255, 0), 2)
        else:
            # img = cv2.ellipse(img, ell, (255, 0, 0), 2)
            print(ell)
            # print(area / img_param, area, img_param, area_param_1, area_param_2)
cv2.imshow("0", img)
cv2.waitKey(0)
