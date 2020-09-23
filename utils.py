import base64
import json

import cv2
import numpy as np
import requests

from cfg import no_found, ATOM_code, CEPH_code, server_ip, callback_interface


def b64string2array(img_string):
    img = np.array([])
    if "base64," in str(img_string):
        img_string = img_string.encode().split(b';base64,')[-1]
    if ".jpg" in str(img_string) or ".png" in str(img_string):
        img_string = img_string.replace("\n", "")
        img = cv2.imread(img_string)
    if len(img_string) > 200:
        img_string = base64.b64decode(img_string)
        np_array = np.frombuffer(img_string, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return img


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
    if response is not None:
        response.raise_for_status()
        result = eval(
            response.content.decode('utf-8').replace(
                'true',
                'True'
            ).replace(
                'false',
                'False'
            ).replace(
                'null',
                'None'
            )
        )
    else:
        result = {'res': response, 'status': 'execution of post failed.'}
    return result


def file_request(function_string, req_id, save_path='Faces_Temp'):
    server_url = server_ip + CEPH_code[function_string]
    if function_string in ['query', 'save']:
        server_url += req_id
        response = requests.post(server_url)
    elif function_string == 'upload':
        assert type(req_id) is dict
        assert 'file' in req_id
        bucket_dict = {'bucketName': 'fries'}
        response = requests.post(server_url, data=bucket_dict, files=req_id)
    else:
        response = {}
    print("Complete post")
    response.raise_for_status()
    try:
        result = eval(
            response.content.decode('utf-8').replace('true', 'True').replace('false', 'False').replace('null', 'None')
        )
    except Exception as e:
        print(repr(e))
        result = {'data': None}
    if function_string == 'query':
        if result['data'] is None:
            return no_found
        file_url = result['data']['server'] + '/' + result['data']['url']
        r = requests.get(file_url)
        with open(save_path + '/' + result['data']['fileName'], 'wb') as f:
            f.write(r.content)
        return result['data']['fileName']
    elif function_string == 'save':
        if result['msg'] == '成功':
            return req_id
        else:
            return -1
    elif function_string == 'upload':
        if result['data'] is not None:
            return result['data']['cephId']
        else:
            return None
    else:
        return None


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