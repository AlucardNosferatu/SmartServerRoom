# import portalocker
import base64
import datetime
import json
import os
import threading
import time
from math import sqrt
from urllib import parse
import cv2
from flask import Flask, request

from VideoTest import camera_async
from cfg_FR import no_found, face_folder_path, save_path
from logger_FR import logger
from utils_FR import process_request, file_request, array2b64string

logger.info('AppFaces starts')
app = Flask(__name__)


@app.route('/imr-ai-service/face_features/check/<file_id>', methods=['POST'])
def check(file_id):
    if request.method == "POST":
        file_id = file_id.replace("\n", "")
        time_take = time.time()

        # f_name = os.path.join('../../Tasks', validate_title('check_' + str(datetime.datetime.now()) + '.json'))
        # with open(f_name, 'w') as task_file:
        #     portalocker.lock(task_file, portalocker.LOCK_EX)
        #     json.dump(
        #         {
        #             'task_id': {
        #                 'main': 'face',
        #                 'sub': 'check'
        #             },
        #             'param_keys': ['file_id'],
        #             'param_dict': {
        #                 'file_id': file_id
        #             }
        #         },
        #         task_file
        #     )

        file_name = file_request(function_string='query', req_id=file_id)
        if file_name == no_found:
            result = -1
            nf = no_found
            logger.error("File not found in CEPH")
        else:
            nf = None
            with open('Faces_Temp/' + file_name, 'rb') as task_file:
                b64_string = base64.b64encode(task_file.read())
                b64_string = b64_string.decode()
                b64_string = 'data:image/jpeg;base64,' + b64_string
            logger.info("Do FD now")
            result = process_request('fd', req_dict={'imgString': b64_string})
            if len(result['res']) > 0:
                logger.info("Result valid")
                new_result = []
                for rect in result['res']:
                    img = cv2.imread('Faces_Temp/' + file_name)
                    # img = img[rect[1]:rect[3], rect[0]:rect[2]]
                    img = img
                    cv2.imwrite('Faces_Temp/cropped_' + file_name, img)
                    with open('Faces_Temp/cropped_' + file_name, 'rb') as fc:
                        b64_string = base64.b64encode(fc.read())
                        b64_string = b64_string.decode()
                        b64_string = 'data:image/jpeg;base64,' + b64_string
                    points = process_request('ld_dbf', req_dict={'imgString': b64_string})
                    r = int(sqrt(img.shape[0] * img.shape[1]) / 200)
                    logger.debug(str(points['res']))
                    for point in points['res']:
                        p = tuple(point)
                        logger.debug(str(p))
                        img = cv2.circle(img=img, center=p, radius=r, color=(255, 0, 0), thickness=-1)
                    cv2.imwrite('Faces_Temp/cropped_' + file_name, img)
                    logger.info("Do upload")
                    uploaded_id = file_request(
                        'upload',
                        {
                            'file': open(
                                'Faces_Temp/cropped_' + file_name,
                                'rb'
                            )
                        }
                    )
                    ret = file_request('save', uploaded_id)
                    if ret == uploaded_id:
                        new_result.append(uploaded_id)
                os.remove('Faces_Temp/cropped_' + file_name)
                new_result = ','.join(new_result)
                result = new_result
                print(new_result)
                logger.debug(str(new_result))
            else:
                result = -1
            os.remove('Faces_Temp/' + file_name)
        time_take = time.time() - time_take

        ret = not (result == -1)
        msg = {True: '成功', False: '失败'}
        if nf is not None:
            result = nf
        return json.dumps(
            {
                'ret': ret,
                'msg': msg[ret],
                'data': result,
                'timeTake': round(time_take, 4)
            },
            ensure_ascii=False
        )


@app.route('/imr-ai-service/face_features/recognize/<file_id>', methods=['POST'])
def recognize(file_id):
    if request.method == "POST":
        file_id = file_id.replace("\n", "")
        time_take = time.time()
        c_da = request.data
        given_c = len(c_da.decode()) > 0
        data = {}
        if given_c:
            try:
                data = eval(c_da.decode())
                given_c = 'x1' in data
                given_c = given_c and 'y1' in data
                given_c = given_c and 'x2' in data
                given_c = given_c and 'y2' in data
            except Exception as e:
                print(repr(e))
                logger.error(repr(e))
                given_c = False
        file_name = file_request(function_string='query', req_id=file_id)
        img = cv2.imread('Faces_Temp/' + file_name)
        if given_c:
            x1 = int(data['x1'].encode().decode())
            y1 = int(data['y1'].encode().decode())
            x2 = int(data['x2'].encode().decode())
            y2 = int(data['y2'].encode().decode())
            result = {'res': [[x1, y1, x2, y2]]}
        else:
            b64str = array2b64string(img).decode()
            # result = process_request('fd_dbf', req_dict={'imgString': b64str})
            result = process_request('fd', req_dict={'imgString': b64str})
        if len(result['res']) > 0:
            img = cv2.imread('Faces_Temp/' + file_name)
            fr_result_list = []
            for index in range(len(result['res'])):
                x1 = result['res'][index][0]
                y1 = result['res'][index][1]
                x2 = result['res'][index][2]
                y2 = result['res'][index][3]
                try:
                    b64str_cropped = array2b64string(img[y1:y2, x1:x2]).decode()
                    fr_result = process_request('fr', req_dict={'imgString': b64str_cropped})
                    fr_result_list.append(fr_result)
                except Exception as e:
                    print(repr(e))
                    logger.error(repr(e))
                    continue
            result = fr_result_list
        else:
            os.remove('Faces_Temp/' + file_name)
            time_take = time.time() - time_take
            return json.dumps(
                {
                    'ret': False,
                    'msg': '失败',
                    'data': '未检测到人脸',
                    'timeTake': round(time_take, 4)
                },
                ensure_ascii=False
            )
        os.remove('Faces_Temp/' + file_name)
        time_take = time.time() - time_take
        return json.dumps(
            {
                'ret': True,
                'msg': '成功',
                'data': result,
                'timeTake': round(time_take, 4)
            },
            ensure_ascii=False
        )


@app.route('/imr-ai-service/face_features/locate/<file_id>', methods=['POST'])
def locate(file_id):
    if request.method == "POST":
        file_id = file_id.replace("\n", "")
        time_take = time.time()

        file_name = file_request(function_string='query', req_id=file_id)
        with open('Faces_Temp/' + file_name, 'rb') as f:
            b64_string = base64.b64encode(f.read())
            b64_string = b64_string.decode()
            b64_string = 'data:image/jpeg;base64,' + b64_string
        result = process_request('fd', req_dict={'imgString': b64_string})
        if len(result['res']) > 0:
            result = result['res']
        else:
            result = -1
        os.remove('Faces_Temp/' + file_name)
        time_take = time.time() - time_take

        ret = not (result == -1)
        msg = {True: '成功', False: '失败'}
        return json.dumps(
            {
                'ret': ret,
                'msg': msg[ret],
                'data': result,
                'timeTake': round(time_take, 4)
            },
            ensure_ascii=False
        )


@app.route('/imr-ai-service/face_features/add/<file_id>', methods=['POST'])
def add(file_id):
    if request.method == "POST":
        file_id = file_id.replace("\n", "")
        time_take = time.time()
        file_name = file_request(function_string='query', req_id=file_id, save_path=face_folder_path)
        result = file_name
        time_take = time.time() - time_take
        if result != -1:
            result = process_request('rr', None)
        ret = not (result == -1)
        msg = {True: '成功', False: '失败'}
        return json.dumps(
            {
                'ret': ret,
                'msg': msg[ret],
                'data': {'reload': result, 'added': file_name},
                'timeTake': round(time_take, 4)
            },
            ensure_ascii=False
        )


@app.route('/imr-ai-service/face_features/delete/<file_name>', methods=['POST'])
def delete(file_name):
    if request.method == "POST":
        file_name = file_name.replace("\n", "")
        time_take = time.time()
        skipped_text = 'reloading was skipped due to an earlier exception.'
        try:
            os.remove(face_folder_path + '/' + file_name)
            reason = None
        except Exception as e:
            reason = repr(e)

        time_take = time.time() - time_take

        if reason is None:
            result = process_request('rr', None)
        else:
            result = skipped_text

        ret = not ((result == -1) or (result == skipped_text))
        msg = {True: '成功', False: '失败'}

        return json.dumps(
            {
                'ret': ret,
                'msg': msg[ret],
                'data': {'reload': result, 'deleted': file_name, 'exception': reason},
                'timeTake': round(time_take, 4)
            },
            ensure_ascii=False
        )


@app.route('/imr-ai-service/face_features/query', methods=['POST'])
def query():
    if request.method == "POST":
        time_take = time.time()
        result = os.listdir(face_folder_path)
        return json.dumps(
            {
                'ret': True,
                'msg': '成功',
                'data': result,
                'timeTake': round(time_take, 4)
            },
            ensure_ascii=False
        )


@app.route('/imr-ai-service/face_features/reload', methods=['POST'])
def reload():
    if request.method == "POST":
        time_take = time.time()

        result = process_request('rr', None)
        return json.dumps(
            {
                'ret': True,
                'msg': '成功',
                'data': result,
                'timeTake': round(time_take, 4)
            },
            ensure_ascii=False
        )


@app.route('/imr-ai-service/face_features/camera_recognize', methods=['POST'])
def camera():
    if request.method == "POST":
        c_da = request.data
        print(str(c_da))
        logger.debug(str(c_da))
        data = json.loads(c_da.decode())
        print(data)
        logger.debug(str(data))
        req_id = data['CameraRecognId']
        rtsp = data['Rtsp_url']
        print(rtsp)
        logger.debug(rtsp)
        sync = data['Asyn']
        try:
            video_type = data['VideoType']
        except Exception as e:
            print(repr(e))
            logger.error(repr(e))
            video_type = None
        try:
            stream_type = data['StreamType']
            assert type(stream_type) is str
        except Exception as e:
            print(repr(e))
            logger.error(repr(e))
            stream_type = '0'
        if type(sync) is str:
            sync = (sync == 'true')
        file_id = data['FileId']
        port = data['RtspPort']
        ch = data['Channel']
        time_take = time.time()
        if file_id is None:
            rtsp = rtsp.replace('+', parse.quote('+'))
            if port is not None:
                rtsp += ':'
                rtsp += port
            if ch is not None:
                if video_type is not None:
                    rtsp += ('/' + video_type)
                    rtsp += ('/ch' + ch)
                    rtsp += ('/' + stream_type)
                    rtsp += '/av_stream'
                else:
                    rtsp += '/cam/realmonitor?channel='
                    rtsp += ch
                    rtsp += '&subtype='
                    rtsp += stream_type
        print("开始截图")
        logger.info("开始截图")

        if sync:
            result = camera_async(
                callbacl_str='camera',
                rtsp=rtsp,
                post_result=False,
                cr_id=req_id,
                count=3,
                wait=60,
                capture=False,
                file_id=file_id
            )
        else:
            t_snap = threading.Thread(
                target=camera_async,
                args=(
                    'camera',
                    rtsp,
                    True,
                    req_id,
                    3,
                    60,
                    False,
                    file_id
                )
            )
            t_snap.start()
            result = None
        time_take = time.time() - time_take
        return json.dumps(
            {
                'code': 1,
                'msg': "请求成功",
                'data': result,
                'timeTake': round(time_take, 4)
            },
            ensure_ascii=False
        )


@app.route('/imr-ai-service/face_features/door_open', methods=['POST'])
def camera2():
    if request.method == "POST":
        print('请求接收时间', str(datetime.datetime.now()))
        logger.debug('请求接收时间' + ' ' + str(datetime.datetime.now()))
        c_da = request.data
        print(str(c_da))
        logger.debug(str(c_da))
        data = json.loads(c_da.decode())
        req_id = data['MediaFileId']
        rtsp = data['Rtsp_url']
        sync = data['Asyn']
        try:
            video_type = data['VideoType']
        except Exception as e:
            print(repr(e))
            logger.error(repr(e))
            video_type = None
        try:
            stream_type = data['StreamType']
            assert type(stream_type) is str
        except Exception as e:
            print(repr(e))
            logger.error(repr(e))
            stream_type = '0'
        if type(sync) is str:
            sync = (sync == 'true')
        file_id = data['FileId']
        port = data['RtspPort']
        ch = data['Channel']
        time_take = time.time()
        if file_id is None:
            rtsp = rtsp.replace('+', parse.quote('+'))
            if port is not None:
                rtsp += ':'
                rtsp += port
            if ch is not None:
                if video_type is not None:
                    rtsp += ('/' + video_type)
                    rtsp += ('/ch' + ch)
                    rtsp += ('/' + stream_type)
                    rtsp += '/av_stream'
                else:
                    rtsp += '/cam/realmonitor?channel='
                    rtsp += ch
                    rtsp += '&subtype='
                    rtsp += stream_type
        print('处理开始时间', str(datetime.datetime.now()))
        logger.debug('处理开始时间' + ' ' + str(datetime.datetime.now()))
        if sync:
            result = camera_async(
                callbacl_str='camera2',
                rtsp=rtsp,
                post_result=False,
                cr_id=req_id,
                count=5,
                wait=1500,
                capture=True,
                file_id=file_id
            )
        else:
            t_snap = threading.Thread(
                target=camera_async,
                args=(
                    'camera2',
                    rtsp,
                    True,
                    req_id,
                    5,
                    1500,
                    True,
                    file_id
                )
            )
            t_snap.start()
            result = None
        time_take = time.time() - time_take
        return json.dumps(
            {
                'code': 1,
                'msg': "请求成功",
                'data': result,
                'timeTake': round(time_take, 4)
            },
            ensure_ascii=False
        )


if __name__ == '__main__':
    pid = os.getpid()
    print('pid is:', pid)
    with open(os.path.join(save_path, 'app_pid.txt'), 'w') as f:
        f.writelines([str(pid)])
    app.run(
        host="0.0.0.0",
        port=int("20291"),
        debug=False, threaded=True
    )
    # file_request('query', '2020082710274358500036c4d1')
