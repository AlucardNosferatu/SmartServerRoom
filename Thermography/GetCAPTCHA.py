import cv2
import numpy as np
import easyocr

reader = easyocr.Reader(['ch_sim', 'en'])  # need to run only once to load model into memory

if __name__ == '__main__':
    path = '../Samples/OCR/1.png'
    img = cv2.imread(path)
    cv2.imshow('CAPTCHA', img)
    cv2.waitKey()
