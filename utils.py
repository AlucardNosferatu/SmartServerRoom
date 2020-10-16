import base64
import json
import re

import cv2
import numpy as np
import requests

from cfg import no_found, ATOM_code, CEPH_code, server_ip, server_ip_2, server_ip_3, callback_interface, download_server


def validate_title(title):
    reg_str = r"[\/\\\:\*\?\"\<\>\|]"  # '/ \ : * ? " < > |'
    new_title = re.sub(reg_str, "_", title)  # 替换为下划线
    return new_title


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


def array2b64string(img_array):
    img_str = cv2.imencode('.jpg', img_array)[1].tostring()  # 将图片编码成流数据，放到内存缓存中，然后转化成string格式
    b64_code = base64.b64encode(img_str)
    return b64_code


def process_request(function_string, req_dict):
    server_url = 'http://127.0.0.1:12241'
    if function_string.endswith('_dbf'):
        server_url = 'http://127.0.0.1:12242'
        function_string = function_string.replace('_dbf', '')
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


def retry_post(server_url, data=None, files=None):
    try:
        if data is None or files is None:
            response = requests.post(server_ip + server_url)
        else:
            response = requests.post(server_ip + server_url, data=data, files=files)
    except Exception as e:
        print(repr(e))
        try:
            if data is None or files is None:
                response = requests.post(server_ip_2 + server_url)
            else:
                response = requests.post(server_ip_2 + server_url, data=data, files=files)
        except Exception as e:
            print(repr(e))
            try:
                if data is None or files is None:
                    response = requests.post(server_ip_3 + server_url)
                else:
                    response = requests.post(server_ip_3 + server_url, data=data, files=files)
            except Exception as e:
                print(repr(e))
                response = None
    return response


def file_request(function_string, req_id, save_path='Faces_Temp'):
    server_url = CEPH_code[function_string]
    if function_string in ['query', 'save']:
        server_url += req_id
        response = retry_post(server_url)
    elif function_string == 'upload':
        assert type(req_id) is dict
        assert 'file' in req_id
        bucket_dict = {'bucketName': 'fries'}
        response = retry_post(server_url, data=bucket_dict, files=req_id)
    else:
        response = None
    print("Complete post")
    if response is None:
        return None
    else:
        response.raise_for_status()
        try:
            result = json.loads(response.content.decode('utf-8'))
        except Exception as e:
            print(repr(e))
            result = {'data': None}
        if function_string == 'query':
            if result['data'] is None:
                return no_found
            file_url = download_server + '/' + result['data']['url']
            r = requests.get(file_url)
            with open(save_path + '/' + result['data']['fileName'], 'wb') as f:
                f.write(r.content)
            return result['data']['fileName']
        elif function_string == 'save':
            if 'msg' in result and result['msg'] == '成功':
                return req_id
            else:
                return None
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
