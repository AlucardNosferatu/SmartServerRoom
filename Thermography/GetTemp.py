import cv2
import numpy as np
# import pytesseract
import easyocr

reader = easyocr.Reader(['ch_sim', 'en'])  # need to run only once to load model into memory


def resize_and_crop(frame):
    frame = cv2.resize(frame, (1024, 768))
    temp_digit = frame[140:180, 880:970, :]
    temp_digit = cv2.resize(temp_digit, (600, 250))
    return temp_digit


def color_filter(frame):
    temp_digit = cv2.convertScaleAbs(frame, alpha=2, beta=-127)
    colorLow = np.array([180, 180, 180])
    colorHigh = np.array([255, 255, 255])
    mask = cv2.inRange(temp_digit, colorLow, colorHigh)
    temp_digit = cv2.cvtColor(cv2.bitwise_and(temp_digit, temp_digit, mask=mask), cv2.COLOR_RGB2GRAY)
    return temp_digit


def img_enhance(frame):
    frame = resize_and_crop(frame)
    frame = color_filter(frame)
    ret_th, temp_digit = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    temp_digit = cv2.morphologyEx(temp_digit, cv2.MORPH_CLOSE, kernel, iterations=5)
    return temp_digit


def process_video(path='Samples/20201010_165228_channelT.dav'):
    cap = cv2.VideoCapture(path)
    count = 0
    while True:
        if count % 1 == 0:
            ret, frame = cap.read()
            if ret:
                temp_digit = img_enhance(frame)
                cv2.imshow('closed', temp_digit)
                result = reader.recognize(temp_digit)
                print(result[0][1])
                cv2.waitKey(1)
            else:
                break
        else:
            cap.grab()
        count += 1


if __name__ == '__main__':
    process_video()
