import cv2

from Obsolete.HOG import detect_and_draw

# sample = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
sample = cv2.VideoCapture('http://admin:admin@192.168.137.31:8081')

while True:
    ret, frame = sample.read()
    if frame is not None:
        frame = cv2.resize(frame, (1024, 768))
        frame = detect_and_draw(frame)
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
    else:
        break
