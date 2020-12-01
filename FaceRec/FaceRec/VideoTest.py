import base64
import datetime
import json
import os
import threading
import time
from math import inf

import cv2
import requests

from cfg_FR import save_path
from logger_FR import logger
from utils_FR import b64string2array, process_request, file_request, response_async, array2b64string, validate_title

detect_dict = {}


def snap(rtsp_address, resize=True, return_multiple=None):
    if '\\' in rtsp_address:
        rtsp_address = rtsp_address.replace('\\', '//')
    if rtsp_address == "LAPTOP_CAMERA":
        cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(rtsp_address)
    if return_multiple is not None:
        print('连续截图进行中...')
        logger.info('连续截图进行中...')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0 or fps == inf:
            fps = 15
        img_str_list = []
        wait = return_multiple[0] * fps
        count = 0
        ret = True
        before = datetime.datetime.now()
        while ret:
            # print('当前流帧序号', count)
            logger.debug('当前流帧序号：' + str(count))
            if len(return_multiple) == 2 and count >= return_multiple[1] * wait:
                print('帧序号超出上限，结束处理')
                logger.info('帧序号超出上限，结束处理')
                break
            if count % wait == 0:
                print('尝试读取并写入，当前流帧序号', count)
                logger.debug('尝试读取并写入，当前流帧序号：' + str(count))
                ret, frame = cap.read()
                if ret:
                    if resize:
                        frame = cv2.resize(frame, (1024, 768))
                    print('把矩阵转换为base64')
                    logger.info('把矩阵转换为base64')
                    b64_code = array2b64string(frame)
                    img_str_list.append(b64_code)
                elif count == 0:
                    print('这流读出来的都是空的啊，是不是RTSP路径有问题？')
                    logger.info('这流读出来的都是空的啊，是不是RTSP路径有问题？')
                    print(rtsp_address)
                    logger.debug(rtsp_address)
                    cap.release()
                    return None
                else:
                    print('读取流失败，结束处理')
                    logger.info('读取流失败，结束处理')
                    cap.release()
                    return img_str_list
                after = datetime.datetime.now()
                print('耗时', str(after - before))
                logger.debug('耗时：' + str(after - before))
                before = after
            else:
                print('跳过该帧，只抓取不解码')
                logger.info('跳过该帧，只抓取不解码')
                time.sleep(0.035)
                cap.grab()
                print('抓取完毕')
                logger.info('抓取完毕')
            count += 1
            print('当前ret', ret)
            logger.debug('当前ret：' + str(ret))
        print('已退出循环')
        logger.info('已退出循环')
        cap.release()
        return img_str_list
    else:
        ret, frame = cap.read()
        if ret:
            if resize:
                frame = cv2.resize(frame, (1024, 768))
            img_str = cv2.imencode('.jpg', frame)[1].tostring()  # 将图片编码成流数据，放到内存缓存中，然后转化成string格式
            b64_code = base64.b64encode(img_str)
            cap.release()
            return b64_code
        else:
            print('这流读出来的都是空的啊，是不是RTSP路径有问题？')
            logger.info('这流读出来的都是空的啊，是不是RTSP路径有问题？')
            print(rtsp_address)
            logger.debug(rtsp_address)
            cap.release()
            return None


def call_recognize(ceph_id):
    server = "http://127.0.0.1:20291"
    url = server + '/imr-ai-service/face_features/recognize/<file_id>'
    url = url.replace('<file_id>', ceph_id)
    headers = {
        "Content-Type": "lication/json; charset=UTF-8"
    }
    response = requests.post(url, headers=headers)
    response.raise_for_status()
    result = json.loads(response.content.decode('utf-8'))
    return result


def loop_until_detected(rtsp, wait, fd_version='fd', prev_cap=None, for_file=False, prev_time=None):
    count = 0
    result = {'res': []}
    img_string = ''
    array = None
    cap_time = datetime.datetime.now()
    if '\\' in rtsp:
        rtsp = rtsp.replace('\\', '//')
    if for_file and prev_cap is not None:
        cap = prev_cap
    else:
        print('RTSP流路径:', rtsp)
        logger.debug('RTSP流路径：' + rtsp)
        if rtsp == "LAPTOP_CAMERA":
            cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
            # cap = cv2.VideoCapture('rtsp://admin:admin@192.168.137.199:8554/live')
        else:
            cap = cv2.VideoCapture(rtsp)

    begin_time = datetime.datetime.now()
    time_elapsed = 0

    span_in_ms = 3000000
    now_is = datetime.datetime.now()
    print('之前是', prev_time)
    logger.debug('之前是：' + str(prev_time))
    print('现在是', now_is)
    logger.debug('现在是：' + str(now_is))
    duration = now_is - prev_time
    rest_time = span_in_ms - duration.microseconds - (1000000 * duration.seconds)
    print('还剩', rest_time)
    logger.debug('还剩：' + str(rest_time))
    while rest_time > 0:
        cap.grab()
        now_is = datetime.datetime.now()
        duration = now_is - prev_time
        rest_time = span_in_ms - duration.microseconds - (1000000 * duration.seconds)
    print('当前时间', str(datetime.datetime.now()))
    logger.debug('当前时间：' + str(datetime.datetime.now()))
    while len(result['res']) == 0 and time_elapsed < wait:
        # print('截三帧', count, time_elapsed)
        # logger.debug('截三帧：' + str(count) + ' ' + str(time_elapsed))
        current_time = datetime.datetime.now()
        time_elapsed = (current_time - begin_time).seconds
        count += 1
        if count % 2 == 0:
            # ss_result = process_request('ss', {'RTSP_ADDR': rtsp})
            cap_time = datetime.datetime.now()
            ret, frame = cap.read()
            # print('截取帧时间', str(cap_time))
            # logger.debug('截取帧时间：' + str(cap_time))
            if ret:
                ss_result = {'result': array2b64string(frame).decode()}
            else:
                ss_result = None
            if type(ss_result) is dict and 'result' in ss_result and ss_result['result'] is not None:
                img_string = ss_result['result']
                array = b64string2array(img_string)
                array = cv2.resize(array, (1024, 768))
                resized_string = array2b64string(array)
                result = process_request(fd_version, req_dict={'imgString': resized_string.decode()})
                print('face_detec_result', result, str(datetime.datetime.now()))
                logger.debug('face_detec_result：' + str(result) + str(datetime.datetime.now()))
            elif not for_file:
                print('snapshot error! reconnect now.')
                logger.info('snapshot error! reconnect now.')
                cap = cv2.VideoCapture(rtsp)
            else:
                print('snapshot error! skip now.')
                logger.info('snapshot error! skip now.')
                return img_string, result, cap, array, cap_time
        else:
            cap.grab()
            # print('跳过当前帧', count)
            # logger.debug('跳过当前帧：' + str(count))
    return img_string, result, cap, array, cap_time


def capture_during_detected(cr_id, rtsp, wait, fd_version='fd', prev_video_w=None, for_file=False, prev_sample=None):
    count = 0
    record_flag = False
    first_time = False
    if '\\' in rtsp:
        rtsp = rtsp.replace('\\', '//')
    if for_file and prev_sample is not None:
        sample = prev_sample
    else:
        first_time = True
        print('RTSP流路径:', rtsp)
        logger.debug('RTSP流路径：' + rtsp)
        if rtsp == "LAPTOP_CAMERA":
            sample = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
        else:
            sample = cv2.VideoCapture(rtsp)
    fps = sample.get(cv2.CAP_PROP_FPS) / 2
    if fps == 0 or fps == inf:
        fps = 12
    print('保存视频FPS:', fps)
    logger.debug('保存视频FPS：' + str(fps))
    fn = validate_title(cr_id)
    output_name = save_path + '/' + fn + '.mp4'
    if prev_video_w is not None and prev_video_w.isOpened():
        video_w = prev_video_w
    else:
        print('write stream using:', output_name)
        logger.debug('write stream using：' + output_name)
        video_w = cv2.VideoWriter(
            output_name,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (512, 384)
        )
    no_face = 0
    before = datetime.datetime.now()
    ret = True
    frame_queue = []
    while count < wait:
        # print('当前帧序号', count)
        # logger.debug('当前帧序号：' + str(count))
        if ret:
            if count % 2 == 0:
                ret, frame = sample.read()
                if ret:
                    frame = cv2.resize(frame, (512, 384))
                    # cv2.imshow('inspection', frame)
                    # cv2.waitKey(1)
                    # print('队列状态', detect_dict[cr_id]['idle'])
                    # logger.debug('队列状态：' + str(detect_dict[cr_id]['idle']))
                    if detect_dict[cr_id]['idle']:
                        result = {'res': detect_dict[cr_id]['rect']}
                        if result['res'] is not None and len(frame_queue) > 0:
                            no_face = 0
                            if not record_flag:
                                record_flag = True
                                count = 0
                            frame_queue[0] = cv2.rectangle(
                                frame_queue[0],
                                (result['res'][0][0], result['res'][0][1]),
                                (result['res'][0][2], result['res'][0][3]),
                                (0, 255, 0),
                                2
                            )
                        elif record_flag:
                            no_face += 1
                        while len(frame_queue) > 0:
                            # print('正在写入流，剩余帧数', len(frame_queue))
                            # logger.debug('正在写入流，剩余帧数：' + str(len(frame_queue)))
                            video_w.write(frame_queue.pop(0))
                        detect_dict[cr_id]['frame'] = frame
                        # 进行异步检测请求
                        print('进行异步检测请求')
                        logger.info('进行异步检测请求')
                        t_detect = threading.Thread(target=detect_async, args=(fd_version, cr_id))
                        t_detect.start()
                    if record_flag or first_time:
                        frame_queue.append(frame)
                else:
                    break
            else:
                sample.grab()
                # print('skip 1 frame')
                # logger.info('skip 1 frame')
        elif not for_file:
            print('连接被切断！现在立刻重连')
            logger.info('连接被切断！现在立刻重连')
            sample = cv2.VideoCapture(rtsp)
        after = datetime.datetime.now()
        # print('消耗时间', str(after - before))
        before = after
        count += 1
    if no_face >= 10:
        record_flag = False
    if for_file:
        pass
    else:
        sample.release()
    return video_w, record_flag, output_name, sample


def crop_and_recognize(img, rect, scene_id, new_result_list):
    # x1 = int(img.shape[1] * rect[0] / 1024)
    # y1 = int(img.shape[0] * rect[1] / 768)
    # x2 = int(img.shape[1] * rect[2] / 1024)
    # y2 = int(img.shape[0] * rect[3] / 768)
    # new_img = img[y1:y2, x1:x2]
    new_img = img
    cv2.imwrite('Faces_Temp/temp.jpg', new_img)
    uploaded_id = file_request('upload', {'file': open('Faces_Temp/temp.jpg', 'rb')})
    if uploaded_id is None:
        return new_result_list
    ret = file_request('save', uploaded_id)
    if ret == uploaded_id:
        result_temp = call_recognize(uploaded_id)['data']
        print('Recognition has been called')
        logger.info('Recognition has been called')
        if len(result_temp) > 0 and 'res' in result_temp[0]:
            for index in range(len(result_temp)):
                print('re-formatting result list')
                logger.info('re-formatting result list')
                print(result_temp[index])
                logger.debug(str(result_temp[index]))
                result_temp[index] = result_temp[index]['res']
                result_temp[index].append(uploaded_id)
                result_temp[index] = {
                    'fileName': str(result_temp[index][0]),
                    'distance': str(result_temp[index][1]),
                    'head_id': str(result_temp[index][2]),
                    'camera': str(scene_id)
                }
                print('result list re-formatted')
                logger.info('result list re-formatted')
                print(result_temp[index])
                logger.debug(str(result_temp[index]))
                new_result_list.append(result_temp[index])
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
    array_list = []
    cap_time = datetime.datetime.now()
    if file_id is not None:
        for_file = True
        file_name = file_request('query', file_id)
        rtsp = os.path.join(save_path, file_name)
    else:
        for_file = False
    while times > 0:
        print('剩余检测次数', times)
        logger.debug('剩余检测次数：' + str(times))
        if capture:
            if record_flag:
                if cr_id not in detect_dict:
                    print('新建状态字典')
                    logger.info('新建状态字典')
                    detect_dict[cr_id] = {'idle': True, 'rect': None, 'frame': None}
                video_w, record_flag, output_name, sample = capture_during_detected(
                    cr_id,
                    rtsp,
                    wait,
                    # fd_version='fd_dbf',
                    fd_version='fd',
                    prev_video_w=video_w,
                    for_file=for_file,
                    prev_sample=sample
                )
                times -= 1
            else:
                times = 0
        else:
            img_string, result, sample, array, cap_time = loop_until_detected(
                rtsp=rtsp,
                wait=wait,
                # fd_version='fd_dbf',
                fd_version='fd',
                prev_cap=sample,
                for_file=for_file,
                prev_time=cap_time
            )
            if post_result:
                array_list.append(array)
                img_string_list.append(img_string)
                box_coordinates.append(result)
                if len(result['res']) == 0:
                    times = 0
                else:
                    times -= 1
            else:
                array_list = [array]
                img_string_list = [img_string]
                box_coordinates = [result]
                times = 0
    if cr_id in detect_dict:
        del detect_dict[cr_id]
    if sample is not None:
        sample.release()
    et = str(datetime.datetime.now())
    if capture and video_w is not None:
        if video_w.isOpened():
            print('release now')
            logger.info('release now')
            video_w.release()
        # 这里做视频上传和保存操作
        f_handle = open(output_name, 'rb')
        video_id = file_request('upload', {'file': f_handle}, bName='inoutmedia')
        f_handle.close()
        print('MediaFileId', video_id)
        logger.debug('MediaFileId：' + str(video_id))
        if video_id is None:
            result = {
                'mediaFileId': cr_id,
                'cephId': None,
                'msg': '失败',
                'status': '上传失败'
            }
        else:
            ret = file_request('save', video_id)
            if ret == video_id:
                data = {'trance_log_id': 'LOCAL_USAGE', 'ceph_id': ret}
                conv_fn = process_request('fc_mdapp', data)
                print('格式转换调用结束')
                logger.info('格式转换调用结束')
                if 'data' in conv_fn and 'ceph_id' in conv_fn['data']:
                    result = {
                        'mediaFileId': cr_id,
                        'cephId': conv_fn['data']['ceph_id'],
                        'msg': '成功',
                        'status': None
                    }
                else:
                    logger.error('格式转换失败，原视频cephId：' + str(ret))
                    result = {
                        'mediaFileId': cr_id,
                        'cephId': None,
                        'msg': '失败',
                        'status': '转换失败'
                    }
                print(result)
                logger.debug(str(result))
            else:
                result = {
                    'mediaFileId': cr_id,
                    'cephId': None,
                    'msg': '失败',
                    'status': '保存失败'
                }
        if os.path.exists(output_name):
            os.remove(output_name)
            print('录制文件已删除')
            logger.info('录制文件已删除')
        if for_file and os.path.exists(rtsp):
            os.remove(rtsp)
            print('转换后文件已删除')
            logger.info('转换后文件已删除')
    else:
        print('截取阶段结束')
        logger.info('截取阶段结束')
        scene_id_list = []
        new_result_list = []
        file_out = os.path.join(save_path, 'scene.jpg')
        for index, img_string in enumerate(img_string_list):
            cv2.imwrite(os.path.join(save_path, 'test_' + str(index) + '.jpg'), array_list[index])
            if len(img_string) <= 0:
                continue
            img = b64string2array(img_string)
            cv2.imwrite(file_out, img)
            scene_id = file_request('upload', {'file': open(file_out, 'rb')})
            print('scene_id', scene_id)
            logger.debug('scene_id：' + str(scene_id))
            if scene_id is None:
                continue
            ret = file_request('save', scene_id)
            if ret == scene_id:
                scene_id_list.append(scene_id)
                os.remove(file_out)
                if len(box_coordinates[index]['res']) > 0 and len(img_string_list) > 1:
                    for rect in box_coordinates[index]['res']:
                        new_result_list = crop_and_recognize(img, rect, scene_id, new_result_list)
        print(new_result_list)
        logger.debug(str(new_result_list))
        result = {
            'cameraRecognId': cr_id,
            'camera': scene_id_list,
            'beginTime': bt,
            'endTime': et,
            'faces': new_result_list
        }
        if len(result['faces']) == 0 and len(result['camera']) > 0:
            result['faces'].append({'camera': result['camera'][0]})
        result['faces'] = remove_duplicated(result['faces'])
        if len(result['camera']) <= 0:
            result['camera'] = None
    print(result)
    logger.debug(str(result))
    response_async(result, callbacl_str)
    try:
        response_async(result, 'listener')
    except Exception as e:
        print(repr(e))
        logger.error(repr(e))
    return result


def snap_per_seconds(rtsp_address, resize, multiple, multiple_mode, data):
    if multiple is not None:
        print('开始进行连续截图')
        logger.info('开始进行连续截图')
    result = snap(rtsp_address=rtsp_address, resize=resize, return_multiple=multiple)
    if multiple_mode and 'upload' in data and data['upload'] is True:
        recode_id = data['recodeId']
        equipment_id = data['equipmentId']
        if result is None:
            result = {
                'cephList': None,
                'recodeId': recode_id,
                'equipmentId': equipment_id,
                'issyn': False,
                'params': None
            }
        else:
            scene_id_list = []
            file_out = os.path.join(save_path, 'scene.jpg')
            for index, img_string in enumerate(result):
                print('该批次第', index, '张准备上传')
                logger.debug('该批次第' + str(index) + '张准备上传')
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
                logger.debug('scene_id：' + str(scene_id))
                if scene_id is None:
                    continue
                else:
                    scene_id_list.append(scene_id)
                if os.path.exists(file_out):
                    os.remove(file_out)
            result = {
                'cephList': scene_id_list,
                'recodeId': recode_id,
                'equipmentId': equipment_id,
                'issyn': False,
                'params': None
            }
        if os.path.exists(rtsp_address):
            os.remove(rtsp_address)
        response_async(result, 'snap')
        try:
            response_async(result, 'listener')
        except Exception as e:
            print(repr(e))
            logger.error(repr(e))
    return result


def detect_async(fd_version, cr_id):
    detect_dict[cr_id]['idle'] = False
    print('开始进行检测', detect_dict[cr_id]['idle'])
    logger.debug('开始进行检测：' + str(detect_dict[cr_id]['idle']))
    frame = detect_dict[cr_id]['frame']
    img_string = array2b64string(frame)
    result = process_request(fd_version, req_dict={'imgString': img_string.decode()})
    if len(result['res']) != 0:
        detect_dict[cr_id]['rect'] = result['res']
    else:
        detect_dict[cr_id]['rect'] = None
    print('检测结束', detect_dict[cr_id]['idle'])
    logger.debug('检测结束：' + str(detect_dict[cr_id]['idle']))
    detect_dict[cr_id]['idle'] = True


def remove_duplicated(faces):
    name_list = {}
    for index, face in enumerate(faces):
        if 'fileName' in face and face['fileName'] != 'no_aligned_faces_detected':
            if face['fileName'] not in name_list:
                name_list[face['fileName']] = face['distance']
            else:
                if name_list[face['fileName']] > face['distance']:
                    name_list[face['fileName']] = face['distance']
    faces = [
        {
            'fileName': face_name,
            'distance': name_list[face_name],
            'head_id': None,
            'camera': None
        } for face_name in name_list
    ]
    if len(faces) <= 0:
        faces = None
    return faces


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
