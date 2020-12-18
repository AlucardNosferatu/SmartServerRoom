import os
import cv2
import numpy as np
import easyocr

reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)  # need to run only once to load model into memory


def enhance(img):
    img = cv2.convertScaleAbs(img, alpha=2, beta=-127)
    ret_th, img_th = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    c = 0.25
    img = cv2.addWeighted(img_th, c, img, 1 - c, 0)
    return img


# 识别函数
def recognize(full_path):
    img = cv2.imread(full_path)
    img = enhance(img)
    result = reader.recognize(img)
    result_str = result[0][1]
    new_str_list = []
    for char in result_str:
        if char.isalnum():
            new_str_list.append(char)
    new_str = ''.join(new_str_list)
    print(new_str)
    cv2.imshow(new_str, cv2.resize(img, (200, 100)))
    return new_str


if __name__ == '__main__':
    # path = '../Samples/OCR'
    # for file in os.listdir(path):
    #     full_path = os.path.join(path, file)
    #     recognize(full_path)
    # cv2.waitKey()
    recognize('../Samples/OCR/1.png')
    cv2.waitKey()
