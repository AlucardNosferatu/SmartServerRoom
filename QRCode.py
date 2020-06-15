import os
import cv2
import copy
import pytesseract
import numpy as np
from ShootingAngle import get_histogram, save_template, cal_cos_similarity

a = 1.3
b = 30
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))


def reshape_image(image):
    '''归一化图片尺寸：短边400，长边不超过800，短边400，长边超过800以长边800为主'''
    width, height = image.shape[1], image.shape[0]
    min_len = width
    scale = width * 1.0 / 800
    new_width = 800
    new_height = int(height / scale)
    if new_height > 1600:
        new_height = 1600
        scale = height * 1.0 / 1600
        new_width = int(width / scale)
    out = cv2.resize(image, (new_width, new_height))
    return out


def enhance(input_image):
    global a
    global b
    h = input_image.shape[0]
    w = input_image.shape[1]
    h2 = int(h / 2)
    w2 = int(w / 2)

    k = cv2.waitKey(20)
    if k & 0xff == ord('q'):
        a += 0.1
        if a == 0:
            a = 0.1
    if k & 0xff == ord('e'):
        a -= 0.1
        if a == 0:
            a = -0.1
    if k & 0xff == ord('a'):
        b += 10
    if k & 0xff == ord('d'):
        b -= 10
    # print(a, "  ", b)
    img_fx = cv2.convertScaleAbs(input_image, alpha=a, beta=b)
    # cv2.imshow("bright", cv2.resize(img_fx, (w2, h2)))
    # cv2.waitKey()
    kernel_sharpen_1 = np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]])
    # 卷积
    img_fx = cv2.filter2D(img_fx, -1, kernel_sharpen_1)
    # cv2.imshow("sharp", cv2.resize(img_fx, (w2, h2)))
    # cv2.waitKey()
    return img_fx


def detect(input_image):
    '''提取所有轮廓'''

    img_s = enhance(input_image)
    h = img_s.shape[0]
    w = img_s.shape[1]
    h2 = int(h / 2)
    w2 = int(w / 2)
    img = np.zeros_like(input_image)
    gray = cv2.cvtColor(img_s, cv2.COLOR_BGR2GRAY)
    _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
    cont, h = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cont, -1, (0, 0, 255), 1)
    cv2.imshow("cont", cv2.resize(img, (w2, h2)))
    # cv2.waitKey()
    return input_image, cont, h


def compute_1(cont, i, j):
    '''最外面的轮廓和子轮廓的比例'''
    area1 = cv2.contourArea(cont[i])
    area2 = cv2.contourArea(cont[j])
    if area2 == 0:
        return False
    ratio = area1 * 1.0 / area2
    if abs(ratio - 49.0 / 25):
        return True
    return False


def compute_2(cont, i, j):
    '''子轮廓和子子轮廓的比例'''
    area1 = cv2.contourArea(cont[i])
    area2 = cv2.contourArea(cont[j])
    if area2 == 0:
        return False
    ratio = area1 * 1.0 / area2
    if abs(ratio - 25.0 / 9):
        return True
    return False


def compute_center(cont, i):
    '''计算轮廓中心点'''
    M = cv2.moments(cont[i])
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return cx, cy


def detect_contours(vec):
    '''判断这个轮廓和它的子轮廓以及子子轮廓的中心的间距是否足够小'''
    distance_1 = np.sqrt((vec[0] - vec[2]) ** 2 + (vec[1] - vec[3]) ** 2)
    distance_2 = np.sqrt((vec[0] - vec[4]) ** 2 + (vec[1] - vec[5]) ** 2)
    distance_3 = np.sqrt((vec[2] - vec[4]) ** 2 + (vec[3] - vec[5]) ** 2)
    if sum((distance_1, distance_2, distance_3)) / 3 < 3:
        return True
    return False


def juge_angle(rec):
    '''判断寻找是否有三个点可以围成等腰直角三角形'''
    if len(rec) < 3:
        return -1, -1, -1
    for i in range(len(rec)):
        for j in range(i + 1, len(rec)):
            for k in range(j + 1, len(rec)):
                distance_1 = np.sqrt((rec[i][0] - rec[j][0]) ** 2 + (rec[i][1] - rec[j][1]) ** 2)
                distance_2 = np.sqrt((rec[i][0] - rec[k][0]) ** 2 + (rec[i][1] - rec[k][1]) ** 2)
                distance_3 = np.sqrt((rec[j][0] - rec[k][0]) ** 2 + (rec[j][1] - rec[k][1]) ** 2)
                if abs(distance_1 - distance_2) < 5:
                    if abs(np.sqrt(np.square(distance_1) + np.square(distance_2)) - distance_3) < 5:
                        return i, j, k
                elif abs(distance_1 - distance_3) < 5:
                    if abs(np.sqrt(np.square(distance_1) + np.square(distance_3)) - distance_2) < 5:
                        return i, j, k
                elif abs(distance_2 - distance_3) < 5:
                    if abs(np.sqrt(np.square(distance_2) + np.square(distance_3)) - distance_1) < 5:
                        return i, j, k
    return -1, -1, -1


def order_points(pts):
    # 初始化坐标点
    rect = np.zeros((4, 2), dtype="float32")

    # 获取左上角和右下角坐标点
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # 分别计算左上角和右下角的离散差值
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def fpt(image, pts):
    # 获取坐标点，并将它们分离开来
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 计算新图片的宽度值，选取水平差值的最大值
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # 计算新图片的高度值，选取垂直差值的最大值
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 构建新图片的4个坐标点
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # 获取仿射变换矩阵并应用它
    M = cv2.getPerspectiveTransform(rect, dst)
    # 进行仿射变换
    warped = cv2.warpPerspective(image, M, (int(maxWidth * 1.85), int(maxHeight * 1.1)))
    warped = warped[:, int(warped.shape[1] * 0.55):]
    # 返回变换后的结果
    return warped


def ocr(warped):
    warped = enhance(warped)
    # warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    cv2.imshow('wrap2', warped)
    # ref = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY)[1]
    # ref = cv2.erode(ref, kernel)
    # ref = cv2.morphologyEx(ref, cv2.MORPH_OPEN, kernel)

    text = pytesseract.image_to_string(warped, lang='chi_sim')
    text = text.replace(' ', '')
    # print(text)
    if "宽带" in text and "联系" in text:
        print("检测成功！！！")

    # cv2.waitKey()


def find(file, img, cont, hier, bf_matcher=None, temp=None, orb=None, root=0):
    '''找到符合要求的轮廓'''
    rec = []
    if not hier.any():
        return
    for i in range(len(hier)):
        child = hier[i][2]
        child_child = hier[child][2]
        if child != -1 and hier[child][2] != -1:
            if compute_1(cont, i, child) and compute_2(cont, child, child_child):
                cx1, cy1 = compute_center(cont, i)
                cx2, cy2 = compute_center(cont, child)
                cx3, cy3 = compute_center(cont, child_child)
                if detect_contours([cx1, cy1, cx2, cy2, cx3, cy3]):
                    rec.append([cx1, cy1, cx2, cy2, cx3, cy3, i, child, child_child])
    '''计算得到所有在比例上符合要求的轮廓中心点'''
    i, j, k = juge_angle(rec)
    if i == -1 or j == -1 or k == -1:
        # print(file, "  ", "Cannot detect QRCode!!!")
        return
    ts = np.concatenate((cont[rec[i][6]], cont[rec[j][6]], cont[rec[k][6]]))
    rect = cv2.minAreaRect(ts)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    result = copy.deepcopy(img)
    warped = fpt(result, box)
    cv2.imshow("warped", warped)
    # cv2.waitKey()
    ocr(warped)
    # kp1, des1 = orb.detectAndCompute(temp, None)
    # kp2, des2 = orb.detectAndCompute(warped, None)
    # matches = bf_matcher.match(des1, des2)
    # matches = sorted(matches, key=lambda x: x.distance)
    # print(len(matches))
    # img2 = cv2.drawMatches(img1=temp, keypoints1=kp1, img2=warped, keypoints2=kp2, matches1to2=matches, outImg=warped,
    #                        flags=2)
    # cv2.imshow("res", img2)
    # cv2.waitKey()
    cv2.drawContours(result, [box], 0, (0, 0, 255), 2)
    cv2.drawContours(img, cont, rec[i][6], (255, 0, 0), 2)
    cv2.drawContours(img, cont, rec[j][6], (255, 0, 0), 2)
    cv2.drawContours(img, cont, rec[k][6], (255, 0, 0), 2)

    height = img.shape[0]
    width = img.shape[1]
    h2 = int(height / 2)
    w2 = int(width / 2)
    cv2.imshow('img-point', cv2.resize(img, (w2, h2)))
    # cv2.waitKey(0)
    # cv2.imshow('img-box', cv2.resize(result, (w2, h2)))
    # cv2.waitKey(0)
    return


if __name__ == '__main__':
    path = "C:\\Users\\16413\\Downloads\\pic(1)"
    sample = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    # template = cv2.imread('Samples\\temp3.jpg', 0)
    # hist_array, img = get_histogram(image=template)
    # save_template(hist_array)
    # temp_hist = hist_array
    # for e, file_name in enumerate(os.listdir(path)):
    file_name = ""
    # bf = cv2.BFMatcher(normType=cv2.NORM_L1, crossCheck=True)
    while sample.isOpened():
        #     orb = cv2.ORB_create()
        ret, image = sample.read()
        if image is not None:
            # if file_name.endswith(".PNG"):
            #     image = cv2.imread(os.path.join(path, file_name))
            image = reshape_image(image)
            height = image.shape[0]
            width = image.shape[1]
            h2 = int(height / 2)
            w2 = int(width / 2)
            cv2.imshow('sb', cv2.resize(image, (w2, h2)))
            # cv2.waitKey(0)
            image, contours, hierarchy = detect(image)
            find(
                # temp=temp_hist,
                # bf_matcher=bf,
                # orb=orb,
                file=file_name,
                img=image,
                cont=contours,
                hier=np.squeeze(hierarchy),
            )
        k = cv2.waitKey(10)
        if k & 0xff == ord('w'):
            break
        # endregion
