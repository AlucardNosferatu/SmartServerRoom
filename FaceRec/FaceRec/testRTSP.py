import cv2
from urllib import parse

r = 'rtsp://admin:admin+@192.168.137.58:8554/live'
r = r.replace('+', parse.quote('+'))
output_name = 'Samples/fuck.mp4'
video_w = cv2.VideoWriter(
    output_name,
    cv2.VideoWriter_fourcc(*'mp4v'),
    25,
    (1024, 768)
)
sample = cv2.VideoCapture(r)
count = 0
while True:
    ret, fr = sample.read()
    if ret:
        fr = cv2.resize(fr, (1024, 768))
        cv2.imshow('rtsp', fr)
        video_w.write(fr)
        count += 1
        cv2.waitKey(1)
    if count > 500:
        break
video_w.release()
sample.release()
cv2.destroyAllWindows()
