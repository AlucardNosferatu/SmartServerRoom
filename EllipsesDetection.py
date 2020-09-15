import math
import cv2
from Sharpen import force_sharpen


def get_ellipse(img):
    img_param = img.shape[0] * img.shape[1]
    blur_param = max(int(img_param / 180000), 1)
    canny_param_2 = int(img_param / 1200)
    canny_param_1 = canny_param_2 + 200
    cnt_length = int(img_param / 2500)
    area_param_1 = int(img_param / 6)
    area_param_2 = int(img_param / 1.5)
    ab_diff_param = max(int(img_param / 35000), 10)
    img = force_sharpen(img)
    img = cv2.blur(img, (blur_param, blur_param))
    cv2.imshow("0", img)

    img_gray = cv2.Canny(img, canny_param_1, canny_param_2)  # Canny边缘检测，参数可更改
    ret, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow("1", thresh)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # contours为轮廓集，可以计算轮廓的长度、面积等
    output_ellipses = []
    for cnt in contours:
        if len(cnt) > cnt_length:
            ell = cv2.fitEllipse(cnt)
            if 0 <= ell[0][0] < img.shape[1] and 0 <= ell[0][1] < img.shape[0]:
                area = math.pi * ell[1][0] * ell[1][1] / 4
                if area_param_1 < area < area_param_2:
                    if abs(ell[1][0] - ell[1][1]) < ab_diff_param:
                        img = cv2.ellipse(img, ell, (0, 255, 0), 2)
                        output_ellipses.append(ell)
                    else:
                        print(abs(ell[1][0] - ell[1][1]))
                        img = cv2.ellipse(img, ell, (255, 0, 0), 1)
                else:
                    img = cv2.ellipse(img, ell, (0, 0, 255), 1)
            else:
                img = cv2.ellipse(img, ell, (0, 255, 255), 1)
    cv2.imshow("2", img)
    # cv2.waitKey(0)
    return output_ellipses


if __name__ == "__main__":
    img = cv2.imread("Samples/watch9.jpg", 3)
    o_e = get_ellipse(img)
