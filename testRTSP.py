import cv2

r = 'rtsp://test:1qaz@134.135.207.73/cam/realmonitor?channel=10&subtype=0'
sample = cv2.VideoCapture(r)
while True:
    ret, fr = sample.read()
    if ret:
        cv2.imshow('rtsp', fr)
        cv2.waitKey(1)
    else:
        break
sample.release()
cv2.destroyAllWindows()
