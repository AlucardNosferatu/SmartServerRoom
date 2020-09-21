import base64
import cv2

# rtsp = 'rtsp://admin:zww123456.@192.168.56.111:5541'


def snap(rtsp_address):
    cap = cv2.VideoCapture(rtsp_address)
    ret, frame = cap.read()
    if frame is not None:
        frame = cv2.resize(frame, (1024, 768))
        img_str = cv2.imencode('.jpg', frame)[1].tostring()  # 将图片编码成流数据，放到内存缓存中，然后转化成string格式
        b64_code = base64.b64encode(img_str)
        return b64_code
    else:
        return None

def async_response(dict):
    pass