import cv2
import numpy as np
# import pytesseract
import easyocr

path = 'Samples/20201010_165228_channelT.dav'

reader = easyocr.Reader(['ch_sim', 'en'])  # need to run only once to load model into memory

cap = cv2.VideoCapture(path)
old_frame = []
diff = []
count = 0
while True:
    if count % 1 == 0:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (1024, 768))
            # cv2.imshow('full', frame)
            temp_digit = frame[140:180, 880:970, :]
            temp_digit = cv2.resize(temp_digit, (600, 250))
            cv2.imshow('color', temp_digit)
            temp_digit = cv2.convertScaleAbs(temp_digit, alpha=2, beta=-127)
            cv2.imshow('gray', temp_digit)
            colorLow = np.array([180, 180, 180])
            colorHigh = np.array([255, 255, 255])
            mask = cv2.inRange(temp_digit, colorLow, colorHigh)
            temp_digit = cv2.cvtColor(cv2.bitwise_and(temp_digit, temp_digit, mask=mask), cv2.COLOR_RGB2GRAY)
            cv2.imshow('mask-plain', temp_digit)
            ret_th, temp_digit = cv2.threshold(temp_digit, 127, 255, cv2.THRESH_BINARY)
            cv2.imshow('thresh', temp_digit)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            temp_digit = cv2.morphologyEx(temp_digit, cv2.MORPH_CLOSE, kernel, iterations=5)
            cv2.imshow('closed', temp_digit)
            result = reader.recognize(temp_digit)
            print(result[0][1])
            cv2.waitKey(1)
        else:
            break
    else:
        cap.grab()
    count += 1
