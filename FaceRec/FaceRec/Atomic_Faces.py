import json
import os
import threading
import time
from urllib import parse

from flask import Flask, request
from UseInsightFace import test_detector, test_recognizer, reload_records
from UseDlib import test_landmarks
from VideoTest import snap_per_seconds
from cfg_FR import save_path
from logger_FR import logger
from utils_FR import b64string2array, file_request

logger.info('AtomicFaces starts')
app = Flask(__name__)


def make_dir(make_dir_path):
    path = make_dir_path.strip()
    if not os.path.exists(path):
        os.makedirs(path)
    return path


@app.route('/test')
def img_start():
    return json.dumps({"system": 0}, ensure_ascii=False)


@app.route('/imr-ai-service/atomic_functions/faces_detect', methods=['POST'])
def faces_detect():
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
        return json.dumps({'res': result, 'timeTake': round(time_take, 4)},
                          ensure_ascii=False)


@app.route('/imr-ai-service/atomic_functions/reload', methods=['POST'])
def reload():
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
    if request.method == "POST":
        c_da = request.data
        data = json.loads(c_da.decode())
        multiple_mode = 'multiple' in data
        multiple_mode = multiple_mode and 'cephId' in data
        multiple_mode = multiple_mode and data['multiple'] is not None
        multiple_mode = multiple_mode and data['cephId'] is not None
        if multiple_mode:
            print('连续截图模式启动.')
            logger.info('连续截图模式启动.')
            multiple = data['multiple']
            file_name = file_request('query', data['cephId'])
            rtsp_address = os.path.join(save_path, file_name)
        else:
            multiple = None
            rtsp_address = data['RTSP_ADDR']
            assert type(rtsp_address) is str
            rtsp_address = rtsp_address.replace('+', parse.quote('+'))
        try:
            resize = data['resize']
        except Exception as e:
            print(repr(e))
            logger.error(repr(e))
            resize = True

        time_take = time.time()
        if multiple_mode:
            t_snap = threading.Thread(
                target=snap_per_seconds,
                args=(
                    rtsp_address,
                    resize,
                    multiple,
                    multiple_mode,
                    data
                )
            )
            t_snap.start()
            result = '异步处理（回调）模式'
            result = result.encode()
        else:
            result = snap_per_seconds(
                rtsp_address=rtsp_address,
                resize=resize,
                multiple=multiple,
                multiple_mode=multiple_mode,
                data=data
            )
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
    pid = os.getpid()
    print('pid is:', pid)
    with open(os.path.join(save_path, 'atomic_pid.txt'), 'w') as f:
        f.writelines([str(pid)])
    app.run(
        host="0.0.0.0",
        port=int("12241"),
        debug=False, threaded=True)
