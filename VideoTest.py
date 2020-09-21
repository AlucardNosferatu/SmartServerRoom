import cv2

rtsp = 'rtsp://admin:zww123456.@192.168.56.111:5541'
cap = cv2.VideoCapture(rtsp)
ret, frame = cap.read()
while ret:
    ret, frame = cap.read()
    if frame is not None:
        frame = cv2.resize(frame, (1024, 768))
        cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()
