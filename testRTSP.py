import cv2

sample = cv2.VideoCapture('rtsp://admin:admin@192.168.137.60:8554/live')
while True:
    ret, fr = sample.read()
    if ret:
        cv2.imshow('rtsp', fr)
        cv2.waitKey(1)
    else:
        break
sample.release()
cv2.destroyAllWindows()
