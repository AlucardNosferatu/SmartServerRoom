import cv2
import numpy as np


def sharpen(img):
    img = enhance(img)
    # img = hist_normalized(img)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], np.float32)  # 定义一个核
    img = cv2.filter2D(img, -1, kernel=kernel)
    return img


def enhance(img):
    # 线性变换
    a = 2
    o = float(a) * img
    o -= 100
    o[o > 255] = 255  # 大于255要截断为255
    o[o < 0] = 0
    # 数据类型的转换
    o = np.round(o)
    o = o.astype(np.uint8)
    return o


def close(img_bin):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img_bin = cv2.dilate(img_bin, kernel)
    img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel)
    return img_bin


def hist_normalized(input_image, O_min=0, O_max=255):
    I_min = np.min(input_image)
    I_max = np.max(input_image)
    rows, cols, _ = input_image.shape
    # 输出图像
    output_image = np.zeros(input_image.shape, np.float32)
    # 输出图像的映射
    cofficient = float(O_max - O_min) / float(I_max - I_min)
    for r in range(rows):
        for c in range(cols):
            output_image[r][c] = cofficient * (input_image[r][c] - I_min) + O_min
    output_image = output_image.astype(np.uint8)
    return output_image


if __name__ == "__main__":
    path = "Samples/watch7.jpg"
    img = cv2.imread(path)
    img = sharpen(img)
    cv2.imshow('fuck', img)
    cv2.waitKey()
