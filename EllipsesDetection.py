import math
import cv2
import numpy as np
from Enhance import sharpen, close


def preprocess(img):
    img_param = img.shape[0] * img.shape[1]
    blur_param = max(int(img_param / 180000), 1)
    canny_param_2 = int(img_param / 1000)
    canny_param_1 = canny_param_2 + 500

    img = sharpen(img)
    # img = cv2.blur(img, (blur_param, blur_param))
    # cv2.imshow("0", img)

    thresh = cv2.Canny(img, canny_param_1, canny_param_2)  # Canny边缘检测，参数可更改
    # cv2.imshow("0.5", thresh)
    thresh = close(thresh)
    # cv2.imshow("1", thresh)
    return thresh, img_param


def get_cont(thresh):
    contours = []
    a = cv2.RETR_TREE
    b1 = cv2.CHAIN_APPROX_NONE
    b2 = cv2.CHAIN_APPROX_SIMPLE
    b3 = cv2.CHAIN_APPROX_TC89_L1
    b4 = cv2.CHAIN_APPROX_TC89_KCOS
    conf = [
        [a, b1],
        [a, b2],
        [a, b3],
        [a, b4]
    ]
    for params in conf:
        cont_temp, hierarchy = cv2.findContours(
            thresh,
            params[0],
            params[1]
        )
        contours += cont_temp
    return contours


def fit_cont(thresh_in, img_in, img_param):
    thresh = thresh_in.copy()
    img = img_in.copy()
    cnt_length = max(int(img_param / 4000), 5)
    area_param_1 = int(img_param * 0.075)
    area_param_2 = int(img_param * 0.9)
    ab_diff_param = 1.2
    contours = get_cont(thresh)  # contours为轮廓集，可以计算轮廓的长度、面积等
    cv2.drawContours(img, contours, -1, (255, 255, 0), 1)
    output_ellipses = []
    for cnt in contours:
        if len(cnt) > cnt_length:
            ell = cv2.fitEllipse(cnt)
            xb1 = int(0.25 * img.shape[1])
            xb2 = int(0.75 * img.shape[1])
            yb1 = int(0.25 * img.shape[0])
            yb2 = int(0.75 * img.shape[0])
            if xb1 <= ell[0][0] < xb2 and yb1 <= ell[0][1] < yb2:
                area = math.pi * ell[1][0] * ell[1][1] / 4
                if area_param_1 < area < area_param_2:
                    if ell[1][0] / ell[1][1] < ab_diff_param and ell[1][1] / ell[1][0] < ab_diff_param:
                        img = cv2.ellipse(img, ell, (0, 255, 0), 5)
                        output_ellipses.append(ell)
                    else:
                        print(abs(ell[1][0] - ell[1][1]))
                        img = cv2.ellipse(img, ell, (255, 0, 0), 1)
                else:
                    img = cv2.ellipse(img, ell, (0, 0, 255), 1)
            else:
                img = cv2.ellipse(img, ell, (0, 255, 255), 1)
    cv2.imshow('2', img)
    cv2.waitKey()
    cv2.waitKey(1)
    return output_ellipses


def get_ellipse(img):
    thresh, img_param = preprocess(img)
    output_ellipses = []
    output_ellipses += fit_cont(thresh, img, img_param)
    return output_ellipses


if __name__ == "__main__":
    test = cv2.imread("Samples/watch1.jpg", 3)
    o_e = get_ellipse(test)
    print(o_e)
    newImg = np.zeros_like(test).astype(np.uint8)
    for ell in o_e:
        newImg = cv2.ellipse(newImg, ell, (255, 255, 255), -1)
        break
    test = cv2.add(test, np.zeros(np.shape(test), dtype=np.uint8), mask=newImg[:, :, 0])
    cv2.imshow('2', test)
    cv2.waitKey()
