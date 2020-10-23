import base64
import datetime
import json
import os
import time
from math import inf

import cv2

# rtsp = 'rtsp://admin:zww123456.@192.168.56.111:5541'
import requests

from cfg_FR import save_path
from utils_FR import b64string2array, process_request, file_request, response_async, array2b64string, validate_title


def snap(rtsp_address, resize=True, return_multiple=None):
    if '\\' in rtsp_address:
        rtsp_address = rtsp_address.replace('\\', '//')
    if rtsp_address == "LAPTOP_CAMERA":
        cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(rtsp_address)
    if return_multiple is not None:
        print('连续截图进行中...')
        img_str_list = []
        wait = return_multiple[0] * 25
        max_count = return_multiple[1] * wait
        count = 0
        ret = True
        while ret:
            print('当前流帧序号', count)
            if count >= max_count:
                break
            if count % wait == 0:
                print('尝试读取并写入，当前流帧序号', count)
                ret, frame = cap.read()
                if ret:
                    if resize:
                        frame = cv2.resize(frame, (1024, 768))
                    img_str = cv2.imencode('.jpg', frame)[1].tostring()  # 将图片编码成流数据，放到内存缓存中，然后转化成string格式
                    b64_code = base64.b64encode(img_str)
                    img_str_list.append(b64_code)
                elif count == 0:
                    print('这流读出来的都是空的啊，是不是RTSP路径有问题？')
                    print(rtsp_address)
                    return None
                else:
                    return img_str_list
            else:
                cap.grab()
            count += 1
        return img_str_list
    else:
        ret, frame = cap.read()
        if ret:
            if resize:
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
        print('截三帧', count)
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


def capture_during_detected(cr_id, rtsp, wait, fd_version='fd', prev_video_w=None, for_file=False, prev_sample=None):
    count = 0
    record_flag = False
    first_time = False
    if '\\' in rtsp:
        rtsp = rtsp.replace('\\', '//')

    if rtsp == "LAPTOP_CAMERA":
        sample = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    else:
        if for_file and prev_sample is not None:
            sample = prev_sample
        else:
            first_time = True
            print('RTSP流路径:', rtsp)
            sample = cv2.VideoCapture(rtsp)

    fps = sample.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps == inf:
        fps = 15
    print('保存视频FPS:', fps)
    fn = validate_title(cr_id)
    output_name = save_path + '/' + fn + '.mp4'
    if prev_video_w is not None and prev_video_w.isOpened():
        video_w = prev_video_w
    else:
        print('write stream using:', output_name)
        video_w = cv2.VideoWriter(
            output_name,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (512, 384)
        )
    no_face = 0
    before = datetime.datetime.now()
    while count < wait:
        count += 1
        ret, frame = sample.read()
        if ret:
            frame = cv2.resize(frame, (512, 384))
            # cv2.imshow('inspection', frame)
            # cv2.waitKey(1)
            if count % 2 == 0:
                img_string = array2b64string(frame)
                result = process_request(fd_version, req_dict={'imgString': img_string.decode()})
                if len(result['res']) != 0:
                    no_face = 0
                    frame = cv2.rectangle(
                        frame,
                        (result['res'][0][0], result['res'][0][1]),
                        (result['res'][0][2], result['res'][0][3]),
                        (0, 255, 0),
                        2
                    )
                    if not record_flag:
                        record_flag = True
                        count = 0
                elif record_flag:
                    no_face += 1
            else:
                print('skip 1 frame')
            if record_flag or first_time:
                print('write now', 'count', count, 'ret', ret, 'ft', first_time, 'rf', record_flag)
                video_w.write(frame)
        elif not for_file:
            print('连接被切断！现在立刻重连')
            sample = cv2.VideoCapture(rtsp)
        after = datetime.datetime.now()
        print('消耗时间', str(after - before))
        before = after
    if no_face >= 10:
        record_flag = False
    if for_file:
        pass
    else:
        sample.release()
    return video_w, record_flag, output_name, sample


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


def camera_async(callbacl_str, rtsp, post_result, cr_id, count=3, wait=25, capture=False, file_id=None):
    bt = str(datetime.datetime.now())
    img_string_list = []
    box_coordinates = []
    output_name = ''
    times = count
    video_w = None
    record_flag = True
    sample = None
    if file_id is not None:
        for_file = True
        file_name = file_request('query', file_id)
        rtsp = os.path.join(save_path, file_name)
    else:
        for_file = False
    while times > 0:
        if capture:
            if record_flag:
                video_w, record_flag, output_name, sample = capture_during_detected(
                    cr_id,
                    rtsp,
                    wait,
                    fd_version='fd_dbf',
                    prev_video_w=video_w,
                    for_file=for_file,
                    prev_sample=sample
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
    if sample is not None:
        sample.release()
    et = str(datetime.datetime.now())

    if capture and video_w is not None:
        if video_w.isOpened():
            print('release now')
            video_w.release()
        # 这里做视频上传和保存操作

        f_handle = open(output_name, 'rb')
        video_id = file_request('upload', {'file': f_handle}, bName='inoutmedia')
        f_handle.close()
        print('MediaFileId', video_id)
        if video_id is None:
            result = {'mediaFileId': cr_id, 'cephId': None, 'msg': '失败', 'status': '上传失败'}
        else:
            ret = file_request('save', video_id)
            if ret == video_id:
                data = {'trance_log_id': 'LOCAL_USAGE', 'ceph_id': ret}
                conv_fn = process_request('fc_mdapp', data)
                if 'data' in conv_fn and 'ceph_id' in conv_fn['data']:
                    result = {'mediaFileId': cr_id, 'cephId': conv_fn['data']['ceph_id'], 'msg': '成功', 'status': None}
                else:
                    result = {'mediaFileId': cr_id, 'cephId': ret, 'msg': '失败', 'status': '转换失败'}
            else:
                result = {'mediaFileId': cr_id, 'cephId': None, 'msg': '失败', 'status': '保存失败'}
        if os.path.exists(output_name):
            os.remove(output_name)
        if for_file and os.path.exists(rtsp):
            os.remove(rtsp)
    else:
        scene_id_list = []
        new_result_list = []
        file_out = os.path.join(save_path, 'scene.jpg')
        for index, img_string in enumerate(img_string_list):
            if len(img_string) <= 0:
                continue
            img = b64string2array(img_string)
            cv2.imwrite(file_out, img)
            scene_id = file_request('upload', {'file': open(file_out, 'rb')})
            print('scene_id', scene_id)
            if scene_id is None:
                continue
            ret = file_request('save', scene_id)
            if ret == scene_id:
                scene_id_list.append(scene_id)
                os.remove(file_out)
                if len(box_coordinates[index]['res']) > 0:
                    for rect in box_coordinates[index]['res']:
                        new_result_list = crop_and_recognize(img, rect, scene_id, new_result_list)
        result = {
            'cameraRecognId': cr_id,
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
    response_async(result, callbacl_str)
    return result


def snap_per_seconds(rtsp_address, resize, multiple, multiple_mode, data):
    print('开始进行连续截图')
    result = snap(rtsp_address=rtsp_address, resize=resize, return_multiple=multiple)
    if multiple_mode and 'upload' in data and data['upload'] is True:
        scene_id_list = []
        file_out = os.path.join(save_path, 'scene.jpg')
        for index, img_string in enumerate(result):
            print()
            if len(img_string) <= 0:
                continue
            img = b64string2array(img_string)
            cv2.imwrite(file_out, img)
            scene_id = file_request(
                'upload',
                {'file': open(file_out, 'rb')},
                bName='inoutmedia'
            )
            print('scene_id', scene_id)
            if scene_id is None:
                continue
            else:
                scene_id_list.append(scene_id)
            if os.path.exists(file_out):
                os.remove(file_out)
        recode_id = data['recodeId']
        equipment_id = data['equipment_id']
        result = {'cephId': scene_id_list, 'recodeId': recode_id, 'equipmentId': equipment_id}
        response_async(result, 'snap')
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
