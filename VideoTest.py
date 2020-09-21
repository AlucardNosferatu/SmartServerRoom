import base64
import os

import cv2


# rtsp = 'rtsp://admin:zww123456.@192.168.56.111:5541'
from App_Faces import process_request, file_request
from Atomic_Faces import b64string2array


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


def response_async(result):
    pass


def camera_async(rtsp,post_result):
    result = {'res': []}
    count = 0
    img_string = ''
    while len(result['res']) == 0:
        img_string = process_request('ss', {'RTSP_ADDR': rtsp})
        result = process_request('fd', req_dict={'imgString': img_string})
        count += 1
    if len(result['res']) > 0:
        img = b64string2array(img_string)
        new_result = []
        for rect in result['res']:
            img = img[rect[1]:rect[3], rect[0]:rect[2]]
            cv2.imwrite('Faces_Temp/temp.jpg', img)
            uploaded_id = file_request(
                'upload',
                {
                    'file': open(
                        'Faces_Temp/temp.jpg',
                        'rb'
                    )
                }
            )
            ret = file_request('save', uploaded_id)
            if ret == uploaded_id:
                new_result.append(uploaded_id)
            os.remove('Faces_Temp/temp.jpg')

        new_result = ','.join(new_result)
        result = new_result
    else:
        result = -1
    if post_result:
        response_async(result)
    return result