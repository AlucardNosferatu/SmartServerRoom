import json
import logging
import os
import time

from flask import Flask, request

from UseDlib import test_detector, test_landmarks, test_recognizer, reload_records
from VideoTest import snap
from utils import b64string2array

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


@app.route('/imr-ai-service/atomic_functions/faces_detect', methods=['POST'])
def faces_detect():
    log_file_name = 'logger-' + time.strftime('%Y-%m-%d', time.localtime(time.time())) + '.log'
    log_file_str = log_file_folder + os.sep + log_file_name
    if not os.path.exists(log_file_str):
        handler = logging.FileHandler(log_file_str, encoding='UTF-8')
        handler.setFormatter(logging_format)
        app.logger.addHandler(handler)
    if request.method == "POST":
        c_da = request.data
        data = json.loads(c_da.decode())
        img_string = data['imgString']
        img = b64string2array(img_string)
        time_take = time.time()
        if img is None:
            result = []
        else:
            result = test_detector(img)
        time_take = time.time() - time_take
        if "fileName" in data.keys():
            app.logger.info("recognition  return:{d},use time:{t}".format(d=result, t=time_take))
            return json.dumps({data['fileName']: [{'value': result}]}, ensure_ascii=False)
        # os.remove(path)
        return json.dumps({'res': result, 'timeTake': round(time_take, 4)},
                          ensure_ascii=False)


@app.route('/imr-ai-service/atomic_functions/landmarks_detect', methods=['POST'])
def landmarks_detect():
    log_file_name = 'logger-' + time.strftime('%Y-%m-%d', time.localtime(time.time())) + '.log'
    log_file_str = log_file_folder + os.sep + log_file_name
    if not os.path.exists(log_file_str):
        handler = logging.FileHandler(log_file_str, encoding='UTF-8')
        handler.setFormatter(logging_format)
        app.logger.addHandler(handler)
    if request.method == "POST":
        c_da = request.data
        data = json.loads(c_da.decode())
        img_string = data['imgString']
        img = b64string2array(img_string)
        time_take = time.time()
        if img is None:
            result = []
        else:
            result = test_landmarks(img)
        time_take = time.time() - time_take
        if "fileName" in data.keys():
            app.logger.info("recognition  return:{d},use time:{t}".format(d=result, t=time_take))
            return json.dumps({data['fileName']: [{'value': result}]}, ensure_ascii=False)
        return json.dumps({'res': result, 'timeTake': round(time_take, 4)},
                          ensure_ascii=False)


@app.route('/imr-ai-service/atomic_functions/recognize', methods=['POST'])
def recognize():
    log_file_name = 'logger-' + time.strftime('%Y-%m-%d', time.localtime(time.time())) + '.log'
    log_file_str = log_file_folder + os.sep + log_file_name
    if not os.path.exists(log_file_str):
        handler = logging.FileHandler(log_file_str, encoding='UTF-8')
        handler.setFormatter(logging_format)
        app.logger.addHandler(handler)
    if request.method == "POST":
        c_da = request.data
        data = json.loads(c_da.decode())
        img_string = data['imgString']
        img = b64string2array(img_string)
        time_take = time.time()
        if img is None:
            result = []
        else:
            result = test_recognizer(img)
        time_take = time.time() - time_take
        if "fileName" in data.keys():
            app.logger.info("recognition  return:{d},use time:{t}".format(d=result, t=time_take))
            return json.dumps({data['fileName']: [{'value': result}]}, ensure_ascii=False)
        return json.dumps({'res': result, 'timeTake': round(time_take, 4)},
                          ensure_ascii=False)


@app.route('/imr-ai-service/atomic_functions/reload', methods=['POST'])
def reload():
    log_file_name = 'logger-' + time.strftime('%Y-%m-%d', time.localtime(time.time())) + '.log'
    log_file_str = log_file_folder + os.sep + log_file_name
    if not os.path.exists(log_file_str):
        handler = logging.FileHandler(log_file_str, encoding='UTF-8')
        handler.setFormatter(logging_format)
        app.logger.addHandler(handler)
    if request.method == "POST":
        time_take = time.time()
        result = reload_records()
        time_take = time.time() - time_take
        return json.dumps(
            {
                'total': result[0],
                'new': result[1],
                'timeTake': round(time_take, 4)
            },
            ensure_ascii=False
        )


@app.route('/imr-ai-service/atomic_functions/snapshot', methods=['POST'])
def snapshot():
    log_file_name = 'logger-' + time.strftime('%Y-%m-%d', time.localtime(time.time())) + '.log'
    log_file_str = log_file_folder + os.sep + log_file_name
    if not os.path.exists(log_file_str):
        handler = logging.FileHandler(log_file_str, encoding='UTF-8')
        handler.setFormatter(logging_format)
        app.logger.addHandler(handler)

    if request.method == "POST":
        c_da = request.data
        data = json.loads(c_da.decode())
        rtsp_address = data['RTSP_ADDR'].encode()
        rtsp_address = rtsp_address.decode()
        time_take = time.time()
        result = snap(rtsp_address)
        time_take = time.time() - time_take
        ret = (result is not None)
        msg = {True: "成功", False: "失败"}
        if ret:
            result = result.decode()
        output = json.dumps(
            {
                'ret': ret,
                'msg': msg[ret],
                'result': result,
                'timeTake': round(time_take, 4)
            },
            ensure_ascii=False
        )
        return output


if __name__ == '__main__':
    app.run(
        host="0.0.0.0",
        port=int("2029"),
        debug=False, threaded=True)
