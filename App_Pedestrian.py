import base64
import json
import logging
import os
import time

import cv2
from flask import Flask, request

from utils import process_request, file_request

app = Flask(__name__)


def make_dir(make_dir_path):
    path = make_dir_path.strip()
    if not os.path.exists(path):
        os.makedirs(path)
    return path


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


@app.route('/imr-ai-service/pedestrian/check/<file_id>', methods=['POST'])
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
                new_result = ','.join(new_result)
                result = new_result
                print(new_result)
            else:
                result = -1
            os.remove('Faces_Temp/' + file_name)
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
        port=int("7120"),
        debug=False, threaded=True)