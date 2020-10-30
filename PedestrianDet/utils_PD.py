import json
import os

import cv2
import base64
import numpy as np
import requests

from cfg_PD import ATOM_code, query_temp_api, query_api, api_server, download_server, save_path, callback_interface


def b64string2array(img_str):
    img = np.array([])
    if "base64," in str(img_str):
        img_str = img_str.split(';base64,')[-1]
    if ".jpg" in str(img_str) or ".png" in str(img_str):
        img_string = img_str.replace("\n", "")
        img = cv2.imread(img_string)
    if len(img_str) > 200:
        img_string = base64.b64decode(img_str)
        np_array = np.fromstring(img_string, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return img


def process_request(function_string, req_dict):
    server_url = 'http://127.0.0.1:12245'
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
    if response is not None:
        response.raise_for_status()
        result = json.loads(response.content.decode('utf-8'))
    else:
        result = {'res': response, 'status': 'execution of post failed.'}
    return result


def download(req_id, from_temp=False):
    if from_temp:
        server_url = query_temp_api
        b_key = 'bucketTemp'
    else:
        server_url = query_api
        b_key = 'bucketName'
    server_url = api_server + server_url + req_id
    response = requests.post(server_url)
    response.raise_for_status()
    result = json.loads(response.content.decode('utf-8'))
    if 'data' not in result or result['data'] is None:
        return '查无此文件'
    file_url = download_server + '/' + result['data'][b_key] + '/' + result['data']['fileName']
    r = requests.get(file_url)
    with open(save_path + '/' + result['data']['fileName'], 'wb') as f:
        f.write(r.content)
    return result['data']['fileName']


def response_async(result, function):
    print("Start to post")
    dic_json = json.dumps(result)
    headers = {
        "Content-Type": "application/json; charset=UTF-8"
    }
    response = requests.post(callback_interface[function], data=dic_json, headers=headers)
    print("Complete post")
    response.raise_for_status()
    print(response.content.decode('utf-8'))


def detect_body_parts(file_id, file_name, recodeId, equipmentId):
    print('写入文件', file_name)
    with open(os.path.join(save_path, file_name), 'rb') as f:
        b64_string = base64.b64encode(f.read())
        b64_string = b64_string.decode()
        b64_string = 'data:image/jpeg;base64,' + b64_string
    print('开始检测')
    result = process_request('pd', req_dict={'imgString': b64_string})
    print(os.path.join(save_path, file_name))
    print(os.path.exists(os.path.join(save_path, file_name)))
    if os.path.exists(os.path.join(save_path, file_name)):
        os.remove(os.path.join(save_path, file_name))
        print('已删除')
    if 'res' in result and 'count' in result['res'] and result['res']['count'] > 0:
        result = {'cephId': file_id}
    else:
        result = {'cephId': None}
    result['recodeId'] = recodeId
    result['equipmentId'] = equipmentId
    response_async(result, 'ped')
    # response_async(result, 'listener')
    return result
