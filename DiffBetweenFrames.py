import cv2
import numpy as np

from HOG import detect_and_draw

sample = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
ret, old_frame = sample.read()
while True:
    ret, frame = sample.read()
    if frame is not None:
        diff_frame = np.abs(frame.astype(np.int8) - old_frame.astype(np.int8)).astype(np.uint8)
        old_frame = frame
        diff_frame = cv2.cvtColor(diff_frame, cv2.COLOR_RGB2GRAY)
        diff_frame = detect_and_draw(diff_frame)
        cv2.imshow('frame', diff_frame)
        cv2.waitKey(1)
    else:
        break
