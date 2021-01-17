import json
import os
import time

import numpy as np
from flask import Flask, request

from logger_MD import logger
from Analysis import get_diff, six_histograms, trigger
from CamMonitor import convert, calc_and_draw_hist
from MostDifferentFrame import snap_atom
from TimeStamp import get_boxes, cut_timestamp
from cfg_MD import save_path
from utils_MD import b64string2array, array2b64string, position_map

app = Flask(__name__)


@app.route('/test')
def img_start():
    return json.dumps({"system": 0}, ensure_ascii=False)


@app.route('/imr-ai-service/atomic_functions/convert', methods=['POST'])
def video_convert():
    if request.method == "POST":
        c_da = request.data
        logger.debug(c_da.decode())
        data = json.loads(c_da.decode())
        video_path = data['file_path']
        video_codec = data['codec']
        video_postfix = data['postfix']
        video_br = data['bitRate']
        video_scale = data['scale']
        video_deletion = data['deletion']
        time_take = time.time()
        if video_path is None:
            result = -1
        else:
            print('start to convert')
            logger.info('start to convert')
            result = convert(
                file_path=video_path,
                codec=video_codec,
                postfix=video_postfix,
                br=video_br,
                new_scale=video_scale
            )
            print('complete convert')
            logger.info('complete convert')
        # if video_deletion and os.path.exists(video_path):
        #     os.remove(video_path)
        time_take = time.time() - time_take
        # os.remove(path)
        return json.dumps({'res': result, 'timeTake': round(time_take, 4)},
                          ensure_ascii=False)


@app.route('/imr-ai-service/atomic_functions/timestamp', methods=['POST'])
def timestamp():
    if request.method == "POST":
        c_da = request.data
        data = json.loads(c_da.decode())
        img_string = data['imgString']
        img = b64string2array(img_string)
        time_take = time.time()
        if img is None:
            result = []
        else:
            result = get_boxes(img)
        time_take = time.time() - time_take
        return json.dumps({'res': result, 'timeTake': round(time_take, 4)},
                          ensure_ascii=False)


@app.route('/imr-ai-service/atomic_functions/cut_a_box', methods=['POST'])
def cut_a_box():
    if request.method == "POST":
        c_da = request.data
        data = json.loads(c_da.decode())
        img_string = data['imgString']
        box = data['boxes']
        img = b64string2array(img_string)
        time_take = time.time()
        if img is None:
            result = []
        else:
            result = cut_timestamp(box, img)
            if type(result) is np.array:
                result = array2b64string(result)
        time_take = time.time() - time_take
        return json.dumps({'res': result, 'timeTake': round(time_take, 4)},
                          ensure_ascii=False)


@app.route('/imr-ai-service/atomic_functions/most_different_frames', methods=['POST'])
def most_different_frames():
    if request.method == "POST":
        c_da = request.data
        data = json.loads(c_da.decode())
        video_path = data['file_path']
        time_take = time.time()
        if video_path is None:
            result = []
        else:
            frame_a, frame_b = snap_atom(calc_and_draw_hist, file_path=video_path)
            if type(frame_a) is np.array and type(frame_b) is np.array:
                frame_a = array2b64string(frame_a)
                frame_b = array2b64string(frame_b)
            result = {'frame_a': frame_a, 'frame_b': frame_b}
        time_take = time.time() - time_take
        return json.dumps({'res': result, 'timeTake': round(time_take, 4)},
                          ensure_ascii=False)


@app.route('/imr-ai-service/atomic_functions/difference_between_frames', methods=['POST'])
def difference_between_frames():
    if request.method == "POST":
        c_da = request.data
        data = json.loads(c_da.decode())
        img_string_a = data['frame_a']
        img_string_b = data['frame_b']
        time_take = time.time()
        if img_string_a is None or img_string_b is None:
            result = []
        else:
            frame_a = b64string2array(img_string_a)
            frame_b = b64string2array(img_string_b)
            frame_a, diff = get_diff(frame_b, frame_a)
            if type(frame_a) is np.array and type(diff) is np.array:
                frame_a = array2b64string(frame_a)
                diff = array2b64string(diff)
            result = {'old_frame': frame_a, 'diff': diff}
        time_take = time.time() - time_take
        return json.dumps({'res': result, 'timeTake': round(time_take, 4)},
                          ensure_ascii=False)


@app.route('/imr-ai-service/atomic_functions/histograms_3x2', methods=['POST'])
def histograms_3x2():
    if request.method == "POST":
        c_da = request.data
        data = json.loads(c_da.decode())
        img_string = data['imgString']
        img = b64string2array(img_string)
        time_take = time.time()
        if img is None:
            result = []
        else:
            size = (img.shape[1], img.shape[0])
            result = six_histograms(img, size)
            result = {
                'left_top': result[0],
                'center_top': result[1],
                'right_top': result[2],
                'left_bottom': result[3],
                'center_bottom': result[4],
                'right_bottom': result[5]
            }
        time_take = time.time() - time_take
        return json.dumps({'res': result, 'timeTake': round(time_take, 4)},
                          ensure_ascii=False)


@app.route('/imr-ai-service/atomic_functions/hot_zone', methods=['POST'])
def hot_zone():
    if request.method == "POST":
        c_da = request.data
        data = json.loads(c_da.decode())
        histograms = data['6histo']
        threshold = data['th']
        time_take = time.time()
        if histograms is None:
            result = []
        else:
            threshold = float(threshold)
            p, sig = trigger(histograms, threshold)
            result = {
                'position': position_map[p],
                'signal_intensity': sig,
                'request_threshold': threshold
            }
        time_take = time.time() - time_take
        return json.dumps({'res': result, 'timeTake': round(time_take, 4)},
                          ensure_ascii=False)


if __name__ == '__main__':
    pid = os.getpid()
    print('pid is:', pid)
    with open(os.path.join(save_path, 'atomic_pid.txt'), 'w') as f:
        f.writelines([str(pid)])
    app.run(
        host="0.0.0.0",
        port=int("12243"),
        debug=False, threaded=True)
