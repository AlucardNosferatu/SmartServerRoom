import json
import os

import cv2
import base64
import numpy as np
import requests

from cfg import callback_interface, ATOM_code, download_server, save_path, server_ip, CEPH_code

position_map = {
    0: 'left_top',
    1: 'center_top',
    2: 'right_top',
    3: 'left_bottom',
    4: 'center_bottom',
    5: 'right_bottom'
}


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


def response_async(result, function, url_param=None):
    print("Start to post")
    dic_json = json.dumps(result)
    headers = {
        "Content-Type": "application/json; charset=UTF-8"
    }
    post_url = callback_interface[function]
    if url_param is not None:
        post_url+=url_param
    response = requests.post(post_url, data=dic_json, headers=headers)
    print("Complete post")
    response.raise_for_status()
    print(response.content.decode('utf-8'))


def process_request(function_string, req_dict):
    server_url = 'http://127.0.0.1:12243'
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
        server_url = CEPH_code['qt']
        b_key = 'bucketTemp'
    else:
        server_url = CEPH_code['query']
        b_key = 'bucket'
    server_url = server_ip + server_url + req_id
    response = requests.post(server_url)
    response.raise_for_status()
    result = json.loads(response.content.decode('utf-8'))
    file_url = download_server + '/' + result['data'][b_key] + '/' + result['data']['fileName']
    r = requests.get(file_url)
    with open(save_path + '/' + result['data']['fileName'], 'wb') as f:
        f.write(r.content)
    return result['data']['fileName']


def upload(file_name, to_temp=False, deletion=True, file_dir=save_path):
    server_url = CEPH_code['upload']
    server_url = server_ip + server_url
    bucket_dict = {'bucketName': 'inoutmedia'}
    file_handle = open(
        os.path.join(file_dir, file_name),
        'rb'
    )
    file_dict = {
        'file': file_handle
    }
    response = requests.post(server_url, data=bucket_dict, files=file_dict)
    response.raise_for_status()
    result = json.loads(response.content.decode('utf-8'))
    result = {'ceph_id': result['data']['cephId']}
    file_handle.close()
    if to_temp:
        result['save_result'] = '上传至bucketTemp'
    else:
        server_url = CEPH_code['save']
        server_url = server_ip + server_url + result['ceph_id']
        response = requests.post(server_url)
        response.raise_for_status()
        save_result = json.loads(response.content.decode('utf-8'))
        result['save_result'] = save_result['msg']
    if deletion and os.path.exists(os.path.join(file_dir, file_name)):
        os.remove(os.path.join(file_dir, file_name))
        result['deletion'] = str(not os.path.exists(os.path.join(file_dir, file_name)))
    else:
        result['deletion'] = '未要求进行删除操作'
    return result
