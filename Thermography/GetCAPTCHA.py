import os
import cv2
import numpy as np
import easyocr

reader = easyocr.Reader(['ch_sim', 'en'])  # need to run only once to load model into memory

if __name__ == '__main__':
    path = '../Samples/OCR'
    for file in os.listdir(path):
        img = cv2.imread(os.path.join(path, file))
        result = reader.recognize(img)
        result_str = result[0][1]
        new_str_list = []
        for char in result_str:
            if char.isalnum():
                new_str_list.append(char)
        new_str = ''.join(new_str_list)
        print(new_str)
        cv2.imshow(new_str, cv2.resize(img, (200, 100)))
    cv2.waitKey()
