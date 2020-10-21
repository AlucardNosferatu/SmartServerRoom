import requests
from flask import Flask, request
import os
import json
import time
import base64
import logging

from cfg_PF import no_found, CEPH_code, ATOM_code, api_server, download_server, qrc_save_path

app = Flask(__name__)


def make_dir(make_dir_path):
    path = make_dir_path.strip()
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def file_request(function_string, req_id, save_path=qrc_save_path):
    server_url = api_server
    server_url += CEPH_code[function_string]
    if function_string in ['query', 'save']:
        server_url += req_id
        response = requests.post(server_url)
    elif function_string == 'upload':
        assert type(req_id) is dict
        assert 'file' in req_id
        bucket_dict = {'bucketName': 'sticker'}
        response = requests.post(server_url, data=bucket_dict, files=req_id)
    else:
        response = {}
    print("Complete post")
    response.raise_for_status()
    result = eval(
        response.content.decode('utf-8').replace('true', 'True').replace('false', 'False').replace('null', 'None')
    )
    if function_string == 'query':
        if result['data'] is None:
            return no_found
        file_url = download_server + '/' + result['data']['url']
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
        return result['data']['cephId']
    else:
        return None


def process_request(function_string, req_dict):
    server_url = 'http://127.0.0.1:12244'
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
    return result


# log init start
log_dir_name = "FaceRec/logs"
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


@app.route('/imr-ai-service/moderm_stickers/check/<file_id>', methods=['POST'])
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
            with open('QRC_Temp/' + file_name, 'rb') as f:
                b64_string = base64.b64encode(f.read())
                b64_string = b64_string.decode()
                b64_string = 'data:image/jpeg;base64,' + b64_string
            try:
                result_1 = process_request('qr', req_dict={'imgString': b64_string})
                result_2 = process_request('dl', req_dict={'imgString': b64_string})
                result = {'qr_evidence': result_1, 'dl_evidence': result_2}
                if result_1['res'].startswith('http://xfujian.189.cn') and result_2['res']['classification'] == '1':
                    result['final_result'] = True
                else:
                    result['final_result'] = False
            except Exception as e:
                print(repr(e))
                result = -1
            os.remove('QRC_Temp/' + file_name)
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


if __name__ == '__main__':
    app.run(
        host="0.0.0.0",
        port=int("20293"),
        debug=False, threaded=True)
