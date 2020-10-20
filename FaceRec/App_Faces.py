import base64
import json
import logging
import os
import threading
import time
from urllib import parse

import cv2
from flask import Flask, request

from VideoTest import camera_async
from utils_FR import process_request, file_request
from cfg_FR import no_found, face_folder_path

app = Flask(__name__)


def make_dir(make_dir_path):
    path = make_dir_path.strip()
    if not os.path.exists(path):
        os.makedirs(path)
    return path


# log init start
log_dir_name = "logs"
log_file_name = 'logger-' + time.strftime('%Y-%m-%d', time.localtime(time.time())) + '.log'
log_file_folder = os.path.join(os.getcwd(), log_dir_name)
make_dir(log_file_folder)
log_file_str = log_file_folder + os.sep + log_file_name
log_level = logging.INFO
handler = logging.FileHandler(log_file_str, encoding='UTF-8')
# handler.setLevel(log_level)
app.logger.setLevel(log_level)
logging_format = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)s - %(message)s')
handler.setFormatter(logging_format)
app.logger.addHandler(handler)


@app.route('/test')
def img_start():
    return json.dumps({"system": 0}, ensure_ascii=False)


@app.route('/imr-ai-service/face_features/check/<file_id>', methods=['POST'])
def check(file_id):
    log_file_name = 'logger-' + time.strftime('%Y-%m-%d', time.localtime(time.time())) + '.log'
    log_file_str = log_file_folder + os.sep + log_file_name
    if not os.path.exists(log_file_str):
        handler = logging.FileHandler(log_file_str, encoding='UTF-8')
        handler.setFormatter(logging_format)
        app.logger.addHandler(handler)

    if request.method == "POST":
        file_id = file_id.replace("\n", "")
        time_take = time.time()

        file_name = file_request(function_string='query', req_id=file_id)
        if file_name == no_found:
            result = -1
            nf = no_found
        else:
            nf = None
            with open('Faces_Temp/' + file_name, 'rb') as f:
                b64_string = base64.b64encode(f.read())
                b64_string = b64_string.decode()
                b64_string = 'data:image/jpeg;base64,' + b64_string
            result = process_request('fd', req_dict={'imgString': b64_string})
            if len(result['res']) > 0:
                new_result = []
                for rect in result['res']:
                    img = cv2.imread('Faces_Temp/' + file_name)
                    img = img[rect[1]:rect[3], rect[0]:rect[2]]
                    cv2.imwrite('Faces_Temp/cropped_' + file_name, img)
                    with open('Faces_Temp/cropped_' + file_name, 'rb') as fc:
                        b64_string = base64.b64encode(fc.read())
                        b64_string = b64_string.decode()
                        b64_string = 'data:image/jpeg;base64,' + b64_string
                    points = process_request('ld', req_dict={'imgString': b64_string})
                    for point in points['res']:
                        img = cv2.circle(img, tuple(point), 2, (255, 0, 0), 1)
                    cv2.imwrite('Faces_Temp/cropped_' + file_name, img)
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
    log_file_name = 'logger-' + time.strftime('%Y-%m-%d', time.localtime(time.time())) + '.log'
    log_file_str = log_file_folder + os.sep + log_file_name
    if not os.path.exists(log_file_str):
        handler = logging.FileHandler(log_file_str, encoding='UTF-8')
        handler.setFormatter(logging_format)
        app.logger.addHandler(handler)

    if request.method == "POST":
        file_id = file_id.replace("\n", "")
        time_take = time.time()
        c_da = request.data

        file_name = file_request(function_string='query', req_id=file_id)
        if file_name == no_found:
            result = -1
            nf = no_found
        else:
            nf = None
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
                    given_c = False
            if given_c:
                x1 = int(data['x1'].encode().decode())
                y1 = int(data['y1'].encode().decode())
                x2 = int(data['x2'].encode().decode())
                y2 = int(data['y2'].encode().decode())
            else:
                with open('Faces_Temp/' + file_name, 'rb') as f:
                    b64_string = base64.b64encode(f.read())
                    b64_string = b64_string.decode()
                    b64_string = 'data:image/jpeg;base64,' + b64_string
                result = process_request('fd', req_dict={'imgString': b64_string})
                if len(result['res']) > 0:
                    area = [(rect[2] - rect[0]) * (rect[3] - rect[1]) for rect in result['res']]
                    index = area.index(max(area))
                    x1 = result['res'][index][0]
                    y1 = result['res'][index][1]
                    x2 = result['res'][index][2]
                    y2 = result['res'][index][3]
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
            img = cv2.imread('Faces_Temp/' + file_name)
            try:
                cv2.imwrite('Faces_Temp/cropped_' + file_name, img[y1:y2, x1:x2])
            except Exception as e:
                print(repr(e))
                os.remove('Faces_Temp/' + file_name)
                time_take = time.time() - time_take
                return json.dumps(
                    {
                        'ret': False,
                        'msg': '失败',
                        'data': '给定坐标超出图片尺寸范围',
                        'timeTake': round(time_take, 4)
                    },
                    ensure_ascii=False
                )

            with open('Faces_Temp/cropped_' + file_name, 'rb') as f:
                b64_string = base64.b64encode(f.read())
                b64_string = b64_string.decode()
                b64_string = 'data:image/jpeg;base64,' + b64_string
            result = process_request('fr', req_dict={'imgString': b64_string})
            os.remove('Faces_Temp/cropped_' + file_name)
            os.remove('Faces_Temp/' + file_name)
        time_take = time.time() - time_take
        if nf is not None:
            result = nf
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
    log_file_name = 'logger-' + time.strftime('%Y-%m-%d', time.localtime(time.time())) + '.log'
    log_file_str = log_file_folder + os.sep + log_file_name
    if not os.path.exists(log_file_str):
        handler = logging.FileHandler(log_file_str, encoding='UTF-8')
        handler.setFormatter(logging_format)
        app.logger.addHandler(handler)

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
    log_file_name = 'logger-' + time.strftime('%Y-%m-%d', time.localtime(time.time())) + '.log'
    log_file_str = log_file_folder + os.sep + log_file_name
    if not os.path.exists(log_file_str):
        handler = logging.FileHandler(log_file_str, encoding='UTF-8')
        handler.setFormatter(logging_format)
        app.logger.addHandler(handler)

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
    log_file_name = 'logger-' + time.strftime('%Y-%m-%d', time.localtime(time.time())) + '.log'
    log_file_str = log_file_folder + os.sep + log_file_name
    if not os.path.exists(log_file_str):
        handler = logging.FileHandler(log_file_str, encoding='UTF-8')
        handler.setFormatter(logging_format)
        app.logger.addHandler(handler)

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
    log_file_name = 'logger-' + time.strftime('%Y-%m-%d', time.localtime(time.time())) + '.log'
    log_file_str = log_file_folder + os.sep + log_file_name
    if not os.path.exists(log_file_str):
        handler = logging.FileHandler(log_file_str, encoding='UTF-8')
        handler.setFormatter(logging_format)
        app.logger.addHandler(handler)

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
    log_file_name = 'logger-' + time.strftime('%Y-%m-%d', time.localtime(time.time())) + '.log'
    log_file_str = log_file_folder + os.sep + log_file_name
    if not os.path.exists(log_file_str):
        handler = logging.FileHandler(log_file_str, encoding='UTF-8')
        handler.setFormatter(logging_format)
        app.logger.addHandler(handler)

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
    log_file_name = 'logger-' + time.strftime('%Y-%m-%d', time.localtime(time.time())) + '.log'
    log_file_str = log_file_folder + os.sep + log_file_name
    if not os.path.exists(log_file_str):
        handler = logging.FileHandler(log_file_str, encoding='UTF-8')
        handler.setFormatter(logging_format)
        app.logger.addHandler(handler)

    if request.method == "POST":
        c_da = request.data
        print(str(c_da))
        data = json.loads(c_da.decode())
        req_id = data['CameraRecognId']
        rtsp = data['Rtsp_url']
        rtsp = rtsp.replace('+', parse.quote('+'))
        sync = data['Asyn']
        try:
            video_type = data['VideoType']
        except Exception as e:
            print(repr(e))
            video_type = None
        try:
            stream_type = data['StreamType']
        except Exception as e:
            print(repr(e))
            stream_type = '0'
        if type(sync) is str:
            sync = (sync == 'true')
        port = data['RtspPort']
        ch = data['Channel']
        time_take = time.time()
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

        if sync:
            result = camera_async('camera', rtsp, False, req_id)
        else:
            t_snap = threading.Thread(target=camera_async, args=('camera', rtsp, True, req_id))
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
    log_file_name = 'logger-' + time.strftime('%Y-%m-%d', time.localtime(time.time())) + '.log'
    log_file_str = log_file_folder + os.sep + log_file_name
    if not os.path.exists(log_file_str):
        handler = logging.FileHandler(log_file_str, encoding='UTF-8')
        handler.setFormatter(logging_format)
        app.logger.addHandler(handler)

    if request.method == "POST":
        c_da = request.data
        print(str(c_da))
        data = json.loads(c_da.decode())
        req_id = data['MediaFileId']
        rtsp = data['Rtsp_url']
        sync = data['Asyn']
        try:
            video_type = data['VideoType']
        except Exception as e:
            print(repr(e))
            video_type = None
        try:
            stream_type = data['StreamType']
        except Exception as e:
            print(repr(e))
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
        if sync:
            result = camera_async('camera2', rtsp, False, req_id, count=5, wait=450, capture=True, file_id=file_id)
        else:
            t_snap = threading.Thread(target=camera_async, args=('camera2', rtsp, True, req_id, 5, 450, True, file_id))
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
    app.run(
        host="0.0.0.0",
        port=int("20291"),
        debug=False, threaded=True)
    # file_request('query', '2020082710274358500036c4d1')
