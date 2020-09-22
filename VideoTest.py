import base64
import datetime
import os

import cv2

# rtsp = 'rtsp://admin:zww123456.@192.168.56.111:5541'
import requests

from utils import b64string2array, process_request, file_request, response_async


def snap(rtsp_address):
    # cap = cv2.VideoCapture(rtsp_address)
    cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    ret, frame = cap.read()
    if frame is not None:
        frame = cv2.resize(frame, (1024, 768))
        img_str = cv2.imencode('.jpg', frame)[1].tostring()  # 将图片编码成流数据，放到内存缓存中，然后转化成string格式
        b64_code = base64.b64encode(img_str)
        return b64_code
    else:
        return None


def call_recognize(ceph_id):
    server = "http://127.0.0.1:7120"
    url = server + '/imr-ai-service/face_features/recognize/<file_id>'
    url = url.replace('<file_id>', ceph_id)
    headers = {
        "Content-Type": "application/json; charset=UTF-8"
    }
    response = requests.post(url, headers=headers)
    response.raise_for_status()
    result = eval(
        response.content.decode('utf-8').replace(
            'true',
            'True'
        ).replace(
            'false',
            'False'
        ).replace(
            'null',
            'None'
        )
    )
    return result


def camera_async(rtsp, post_result, cr_id):
    result = {'res': []}
    count = 0
    img_string = ''
    bt = str(datetime.datetime.now())

    img_string_list = []
    times = 3
    while times > 0:
        while len(result['res']) == 0 and count < 60:
            count += 1
            ss_result = process_request('ss', {'RTSP_ADDR': rtsp})
            if type(ss_result) is dict and 'result' in ss_result and ss_result['result'] is not None:
                img_string = ss_result['result']
                result = process_request('fd', req_dict={'imgString': img_string})
            else:
                continue
        if post_result:
            img_string_list.append(img_string)
            times -= 1
        else:
            img_string_list = [img_string]

    et = str(datetime.datetime.now())

    scene_id_list = []
    new_result_list = []

    for img_string in img_string_list:
        img = b64string2array(img_string)
        cv2.imwrite('Faces_Temp/scene.jpg', img)
        scene_id = file_request('upload', {'file': open('Faces_Temp/scene.jpg', 'rb')})
        ret = file_request('save', scene_id)
        if ret == scene_id:
            scene_id_list.append(scene_id)
            os.remove('Faces_Temp/scene.jpg')
            if len(result['res']) > 0:
                new_result = []
                for rect in result['res']:
                    new_img = img[rect[1]:rect[3], rect[0]:rect[2]]
                    cv2.imwrite('Faces_Temp/temp.jpg', new_img)
                    uploaded_id = file_request('upload', {'file': open('Faces_Temp/temp.jpg', 'rb')})
                    ret = file_request('save', uploaded_id)
                    if ret == uploaded_id:
                        result_temp = call_recognize(uploaded_id)['data']['res']
                        result_temp.append(uploaded_id)
                        new_result.append(result_temp)
                    os.remove('Faces_Temp/temp.jpg')
                new_result_list.append(new_result)
    result = {
        'CameraRecognId': cr_id,
        'camera': scene_id_list,
        'beginTime': bt,
        'endTime': et,
        'faces': new_result_list
    }
    if post_result:
        response_async(result, 'camera')
    return result


if __name__ == '__main__':
    pass
