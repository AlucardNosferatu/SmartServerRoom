import base64
import json
import logging
import os
import time

import cv2
import requests
from flask import Flask, request

# CEPH_code = {
#     'query': '/imr-ceph-server/ceph/query/',
#     'upload': '/imr-ceph-server/ceph/upload/',
#     'save': '/imr-ceph-server/ceph/save/'
# }
CEPH_code = {
    'query': '/imr-ceph-server/ceph/query/',
    'upload': '/imr-ceph-server/ceph/upload/',
    'save': '/imr-ceph-server/ceph/save/'
}
ATOM_code = {
    'fd': '/imr-ai-service/atomic_functions/faces_detect',
    'ld': '/imr-ai-service/atomic_functions/landmarks_detect',
    'fr': '/imr-ai-service/atomic_functions/recognize',
    'rr': '/imr-ai-service/atomic_functions/reload',
}
app = Flask(__name__)


def make_dir(make_dir_path):
    path = make_dir_path.strip()
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def file_request(function_string, req_id):
    # server_url = 'http://134.134.13.81:8788'
    server_url = 'http://192.168.14.212:29999'
    server_url += CEPH_code[function_string]
    if function_string in ['query', 'save']:
        server_url += req_id
        response = requests.post(server_url)
    elif function_string == 'upload':
        assert type(req_id) is dict
        assert 'file' in req_id
        bucket_dict = {'bucketName': 'face'}
        response = requests.post(server_url, data=bucket_dict, files=req_id)
    else:
        response = {}
    print("Complete post")
    response.raise_for_status()
    result = eval(
        response.content.decode('utf-8').replace('true', 'True').replace('false', 'False').replace('null', 'None')
    )
    if function_string == 'query':
        file_url = result['data']['server'] + '/' + result['data']['url']
        r = requests.get(file_url)
        with open('Faces_Temp/' + result['data']['fileName'], 'wb') as f:
            f.write(r.content)
        return result['data']['fileName']
    elif function_string == 'save':
        if result['msg'] == '成功':
            return req_id
        else:
            return -1
    elif function_string == 'upload':
        return result['data']['cephId']
    else:
        return None


def process_request(function_string, req_dict):
    server_url = 'http://127.0.0.1:2029'
    server_url += ATOM_code[function_string]
    headers = {
        "Content-Type": "application/json; charset=UTF-8"
    }
    if req_dict is not None:
        json_dict = json.dumps(req_dict)
        response = requests.post(server_url, data=json_dict, headers=headers)
    else:
        response = requests.post(server_url, headers=headers)
    print("Complete post")
    response.raise_for_status()
    result = eval(response.content.decode('utf-8').replace('true', 'True'))
    return result


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
            result = new_result
        else:
            result = -1
        os.remove('Faces_Temp/' + file_name)
        time_take = time.time() - time_take

        ret = not (result == -1)
        msg = {True: '成功', False: '未检测到人脸'}
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

        data = eval(c_da.decode())
        x1 = int(data['x1'].encode().decode())
        y1 = int(data['y1'].encode().decode())
        x2 = int(data['x2'].encode().decode())
        y2 = int(data['y2'].encode().decode())

        file_name = file_request(function_string='query', req_id=file_id)

        cv2.imwrite('Faces_Temp/cropped_' + file_name, cv2.imread('Faces_Temp/' + file_name)[y1:y2, x1:x2])

        with open('Faces_Temp/cropped_' + file_name, 'rb') as f:
            b64_string = base64.b64encode(f.read())
            b64_string = b64_string.decode()
            b64_string = 'data:image/jpeg;base64,' + b64_string
        result = process_request('fr', req_dict={'imgString': b64_string})
        os.remove('Faces_Temp/cropped_' + file_name)
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


if __name__ == '__main__':
    app.run(
        host="0.0.0.0",
        port=int("7120"),
        debug=False, threaded=True)
    # file_request('query', '2020082710274358500036c4d1')
