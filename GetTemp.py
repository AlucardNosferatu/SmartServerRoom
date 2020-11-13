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
    if count % 2 == 0:
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = cv2.resize(frame, (1024, 768))
            temp_digit = frame[140:180, 880:970]
            # temp_digit = cv2.resize(temp_digit, (600, 250))
            temp_digit = cv2.convertScaleAbs(temp_digit, alpha=1.2, beta=-50)
            cv2.imshow('gray', temp_digit)
            ret_th, thresh = cv2.threshold(temp_digit, 127, 255, cv2.THRESH_BINARY)
            cv2.imshow('thresh', thresh)
            if len(old_frame) < 10:
                diff.append(thresh.copy())
                old_frame.append(thresh.copy())
            else:
                for i in range(len(old_frame)):
                    diff[i] = thresh - old_frame[i]
                old_frame.pop(0)
                old_frame.append(thresh.copy())
            location = np.ones_like(thresh) == 0
            for i in range(len(diff)):
                location = np.logical_or(location, np.abs(diff[i]) > 127)
            thresh[np.where(location)] = 0
            # result = pytesseract.image_to_string(thresh)
            result = reader.recognize(thresh)
            print(result)
            cv2.imshow('Thermal', thresh)
            cv2.waitKey(1)
        else:
            break
    else:
        cap.grab()
    count += 1
