import base64
import json
import logging
import os
import time

import requests
from flask import Flask, request

CEPH_code = {'query': '/imr-ceph-server/ceph/query/'}
ATOM_code = {'fd': '/faces_detect'}
app = Flask(__name__)


def make_dir(make_dir_path):
    path = make_dir_path.strip()
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def file_request(function_string, req_id):
    server_url = 'http://134.134.13.81:8788'
    server_url += CEPH_code[function_string]
    server_url += req_id
    headers = {
        "Content-Type": "application/json; charset=UTF-8"
    }
    response = requests.post(server_url, headers=headers)
    print("Complete post")
    response.raise_for_status()
    result = eval(response.content.decode('utf-8').replace('true', 'True'))
    if function_string == 'query':
        file_url = result['data']['server'] + '/' + result['data']['url']
        r = requests.get(file_url)
        with open('Faces_Temp/' + result['data']['fileName'], 'wb') as f:
            f.write(r.content)
        return result['data']['fileName']


def process_request(function_string, req_dict):
    server_url = 'http://127.0.0.1:2029'
    server_url += ATOM_code[function_string]
    headers = {
        "Content-Type": "application/json; charset=UTF-8"
    }
    json_dict = json.dumps(req_dict)
    response = requests.post(server_url, data=json_dict, headers=headers)
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


@app.route('/imr-face-service/face_collection/checkimage/<file_id>', methods=['POST'])
def img_test(file_id):
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
        os.remove('Faces_Temp/' + file_name)
        time_take = time.time() - time_take

        # os.remove(path)
        return json.dumps({'res': result, 'timeTake': round(time_take, 4)},
                          ensure_ascii=False)


if __name__ == '__main__':
    app.run(
        host="0.0.0.0",
        port=int("7120"),
        debug=False, threaded=True)
    # file_request('query', '2020082710274358500036c4d1')
