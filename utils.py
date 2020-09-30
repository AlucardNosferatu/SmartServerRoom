import cv2
import base64
import numpy as np


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
        result = json.loads(response.content.decode('utf-8'))
    else:
        result = {'res': response, 'status': 'execution of post failed.'}
    return result