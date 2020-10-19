import json

import cv2
import base64
import numpy as np
import requests

from cfg_PD import ATOM_code, query_temp_api, query_api, api_server, download_server, save_path


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
    file_url = download_server + '/' + result['data'][b_key] + '/' + result['data']['fileName']
    r = requests.get(file_url)
    with open(save_path + '/' + result['data']['fileName'], 'wb') as f:
        f.write(r.content)
    return result['data']['fileName']
