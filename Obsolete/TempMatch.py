import os
import cv2
import numpy as np


def change_size(image):
    # image = cv2.medianBlur(image, 5)  # 中值滤波，去除黑色边际中可能含有的噪声干扰
    b = cv2.threshold(image, 15, 255, cv2.THRESH_BINARY)  # 调整裁剪效果
    binary_image = b[1]  # 二值图--具有三通道
    # binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    # print(binary_image.shape)  # 改为单通道

    x = binary_image.shape[0]
    # print("高度x=", x)
    y = binary_image.shape[1]
    # print("宽度y=", y)
    fast_process = list(np.where(binary_image == 0))

    edges_x = fast_process[0]
    edges_y = fast_process[1]
    # for i in range(x):
    #     for j in range(y):
    #         if binary_image[i, j] == 255:
    #             edges_x.append(i)
    #             edges_y.append(j)

    left = min(edges_x)  # 左边界
    right = max(edges_x)  # 右边界
    width = right - left  # 宽度
    bottom = min(edges_y)  # 底部
    top = max(edges_y)  # 顶部
    height = top - bottom  # 高度

    pre1_picture = image[left:left + width, bottom:bottom + height]  # 图片截取
    return pre1_picture


def rotate_bound(image, angle):
    # 获取图像的尺寸
    # 旋转中心
    (h, w) = image.shape[:2]
    (cx, cy) = (w / 2, h / 2)
    # 设置旋转矩阵
    M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # 计算图像旋转后的新边界
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # 调整旋转矩阵的移动距离（t_{x}, t_{y}）
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    return cv2.warpAffine(image, M, (nW, nH))


def naive_matcher():
    # img = cv2.imread('Samples\\LMS\\6.PNG', 0)
    template = cv2.imread('..\\Samples\\temp2.jpg', 1)
    path = "..\\Samples\\LMS"
    # orb = cv2.ORB_create(nfeatures=20)
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=20)
    FLANN_INDEX_KDTREE = 0
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    searchParams = dict(checks=50)
    for e, file_name in enumerate(os.listdir(path)):
        if file_name.endswith(".PNG"):
            img = cv2.imread(os.path.join(path, file_name))
            for i in range(0, 360, 2):
                # print(i)
                rot = rotate_bound(template, i)
                rot = change_size(rot)
                cv2.imshow("rot", rot)
                # kp1, des1 = orb.detectAndCompute(rot, None)
                # kp2, des2 = orb.detectAndCompute(img, None)
                kp1, des1 = sift.detectAndCompute(rot, None)
                kp2, des2 = sift.detectAndCompute(img, None)
                # bf = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=True)
                flann = cv2.FlannBasedMatcher(indexParams, searchParams)

                # matches = bf.match(des1, des2)
                # matches = sorted(matches, key=lambda x: x.distance)
                # img2 = cv2.drawMatches(
                #     img1=rot,
                #     keypoints1=kp1,
                #     img2=img,
                #     keypoints2=kp2,
                #     matches1to2=matches,
                #     outImg=img,
                #     flags=2
                # )

                matches = flann.knnMatch(des1, des2, k=2)
                matches_mask = [[0, 0] for i in range(len(matches))]
                for i, (m, n) in enumerate(matches):
                    if m.distance < 0.7 * n.distance:
                        matches_mask[i] = [1, 0]
                draw_params = dict(
                    matchColor=(0, 255, 0),
                    singlePointColor=(255, 0, 0),
                    matchesMask=matches_mask,
                    flags=0
                )
                img2 = cv2.drawMatchesKnn(rot, kp1, img, kp2, matches, None, **draw_params)

                # region Match template
                # h, w = rot.shape[:2]
                # res = cv2.matchTemplate(img, rot, cv2.TM_SQDIFF_NORMED)
                # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                # left_top = max_loc  # 左上角
                # right_bottom = (left_top[0] + w, left_top[1] + h)  # 右下角
                # img_copy = img.copy()
                # cv2.rectangle(img_copy, left_top, right_bottom, 255, 2)  # 画出矩形位置
                # cv2.imshow("img", img_copy)
                # endregion
                cv2.imshow("res", img2)
                cv2.waitKey()
                cv2.waitKey(1)


if __name__ == '__main__':
    naive_matcher()
