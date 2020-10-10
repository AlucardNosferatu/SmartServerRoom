import base64
import datetime
import json
import os
import time
from math import inf

import cv2

# rtsp = 'rtsp://admin:zww123456.@192.168.56.111:5541'
import requests

from cfg import save_path
from utils import b64string2array, process_request, file_request, response_async, array2b64string


def snap(rtsp_address):
    if '\\' in rtsp_address:
        rtsp_address = rtsp_address.replace('\\', '//')
    if rtsp_address == "LAPTOP_CAMERA":
        cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(rtsp_address)
    ret, frame = cap.read()
    if frame is not None:
        frame = cv2.resize(frame, (1024, 768))
        img_str = cv2.imencode('.jpg', frame)[1].tostring()  # 将图片编码成流数据，放到内存缓存中，然后转化成string格式
        b64_code = base64.b64encode(img_str)
        return b64_code
    else:
        print('这流读出来的都是空的啊，是不是RTSP路径有问题？')
        print(rtsp_address)
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
    result = json.loads(response.content.decode('utf-8'))
    return result


def loop_until_detected(rtsp, wait, fd_version='fd'):
    count = 0
    result = {'res': []}
    img_string = ''
    while len(result['res']) == 0 and count < wait:
        print(count)
        count += 1
        ss_result = process_request('ss', {'RTSP_ADDR': rtsp})
        if type(ss_result) is dict and 'result' in ss_result and ss_result['result'] is not None:
            img_string = ss_result['result']
            result = process_request(fd_version, req_dict={'imgString': img_string})
            print('face_detec_result', result)
        else:
            print('snapshot error! skip now.')
            return img_string, result
    return img_string, result


def capture_during_detected(rtsp, wait, fd_version='fd', prev_video_w=None):
    count = 0
    record_flag = False
    result = {'res': []}

    if '\\' in rtsp:
        rtsp = rtsp.replace('\\', '//')
    if rtsp == "LAPTOP_CAMERA":
        sample = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    else:
        sample = cv2.VideoCapture(rtsp)
    fps = sample.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps == inf:
        fps = 15
    size = (
        int(sample.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(sample.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )
    output_name = save_path + '/temp.mp4'
    if prev_video_w is not None and prev_video_w.isOpened():
        video_w = prev_video_w
    else:
        video_w = cv2.VideoWriter(
            output_name,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            size
        )
    while count < wait:
        print(count)
        count += 1
        ret, frame = sample.read()
        if frame is not None and not record_flag:
            img_string = array2b64string(frame)
            result = process_request(fd_version, req_dict={'imgString': img_string.decode()})
            if len(result['res']) != 0:
                record_flag = True
                count = 0
        if record_flag:
            video_w.write(frame)
    sample.release()
    return video_w, record_flag, output_name


def crop_and_recognize(img, rect, scene_id, new_result_list):
    new_img = img[rect[1]:rect[3], rect[0]:rect[2]]
    cv2.imwrite('Faces_Temp/temp.jpg', new_img)
    uploaded_id = file_request('upload', {'file': open('Faces_Temp/temp.jpg', 'rb')})
    if uploaded_id is None:
        return new_result_list
    ret = file_request('save', uploaded_id)
    if ret == uploaded_id:
        result_temp = call_recognize(uploaded_id)['data']
        if 'res' in result_temp:
            result_temp = result_temp['res']
            result_temp.append(uploaded_id)
            result_temp = {
                'fileName': str(result_temp[0]),
                'distance': str(result_temp[1]),
                'head_id': str(result_temp[2]),
                'camera': str(scene_id)
            }
            new_result_list.append(result_temp)
    os.remove('Faces_Temp/temp.jpg')
    return new_result_list


def camera_async(rtsp, post_result, cr_id, count=3, wait=25, capture=False):
    bt = str(datetime.datetime.now())
    img_string_list = []
    box_coordinates = []
    output_name = ''
    times = count
    video_w = None
    record_flag = True
    while times > 0:
        if capture:
            if record_flag:
                video_w, record_flag, output_name = capture_during_detected(
                    rtsp,
                    wait,
                    fd_version='fd',
                    prev_video_w=video_w
                )
                times -= 1
            else:
                times = 0
        else:
            img_string, result = loop_until_detected(rtsp, wait, fd_version='fd_dbf')
            if post_result:
                img_string_list.append(img_string)
                box_coordinates.append(result)
                if len(result['res']) == 0:
                    times = 0
                else:
                    print('sleep now')
                    times -= 1
                    time.sleep(1)
                    print('awake now')
            else:
                img_string_list = [img_string]
                box_coordinates = [result]
                times = 0

    et = str(datetime.datetime.now())

    if capture and video_w is not None:
        if video_w.isOpened():
            video_w.release()
        # 这里做视频上传和保存操作
        video_id = file_request('upload', {'file': open(output_name, 'rb')})
        print('video_id', video_id)
        if video_id is None:
            result = {'ceph_id': None, 'msg': '失败', 'status': '上传失败'}
        else:
            ret = file_request('save', video_id)
            if ret == video_id:
                result = {'ceph_id': video_id, 'msg': '成功', 'status': None}
            else:
                result = {'ceph_id': None, 'msg': '失败', 'status': '保存失败'}
    else:
        scene_id_list = []
        new_result_list = []
        for index, img_string in enumerate(img_string_list):
            if len(img_string) <= 0:
                continue
            img = b64string2array(img_string)
            cv2.imwrite('Faces_Temp/scene.jpg', img)
            scene_id = file_request('upload', {'file': open('Faces_Temp/scene.jpg', 'rb')})
            print('scene_id', scene_id)
            if scene_id is None:
                continue
            ret = file_request('save', scene_id)
            if ret == scene_id:
                scene_id_list.append(scene_id)
                os.remove('Faces_Temp/scene.jpg')
                if len(box_coordinates[index]['res']) > 0:
                    for rect in box_coordinates[index]['res']:
                        new_result_list = crop_and_recognize(img, rect, scene_id, new_result_list)
        result = {
            'CameraRecognId': cr_id,
            'camera': scene_id_list,
            'beginTime': bt,
            'endTime': et,
            'faces': new_result_list
        }
        if len(result['faces']) == 0 and len(result['camera']) > 0:
            result['faces'].append({'camera': result['camera'][0]})
    print('')
    print('')
    print(result)
    print('')
    print('')
    response_async(result, 'camera')
    return result


if __name__ == '__main__':
    cap = cv2.VideoCapture('rtsp://admin:zww123456.@192.168.56.111:5542')
    while True:
        ret, frame = cap.read()
        if frame is not None:
            frame = cv2.resize(frame, (1024, 768))
            cv2.imshow('fuck', frame)
            cv2.waitKey(1)
        else:
            break
