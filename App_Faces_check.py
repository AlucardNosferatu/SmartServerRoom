import requests
from flask import Flask, request
import os
import cv2
import json
import time

import base64
from skimage import io
import numpy as np
import logging

from UseDlib import test_detector

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
        with open('Face_Temp/' + result['data']['fileName'], 'wb') as f:
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


@app.route('/check', methods=['POST'])
def img_test():
    log_file_name = 'logger-' + time.strftime('%Y-%m-%d', time.localtime(time.time())) + '.log'
    log_file_str = log_file_folder + os.sep + log_file_name
    if not os.path.exists(log_file_str):
        handler = logging.FileHandler(log_file_str, encoding='UTF-8')
        handler.setFormatter(logging_format)
        app.logger.addHandler(handler)

    if request.method == "POST":
        start_time = time.time()
        c_da = request.data
        data = str(c_da, "utf-8")

        data = eval(c_da.decode())
        img_string = data['imgString'].encode()
        img_string = img_string.decode()
        # print(imgString)
        img = np.array([])

        app.logger.info(data)
        img_string = img_string.replace("\n", "")

        img_rgb = io.imread(img_string)
        img = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)

        time_take = time.time()

        result = test_detector(img)

        time_take = time.time() - time_take

        if "fileName" in data.keys():
            app.logger.info("recognition  return:{d},use time:{t}".format(d=result, t=time_take))
            return json.dumps({data['fileName']: [{'value': result}]}, ensure_ascii=False)
        # os.remove(path)
        return json.dumps({'res': result, 'timeTake': round(time_take, 4)},
                          ensure_ascii=False)


if __name__ == '__main__':
    # app.run(
    #     host="0.0.0.0",
    #     port=int("7120"),
    #     debug=False, threaded=True)
    file_request('query', '2020082710274358500036c4d1')
