import json
import os
import threading
import time

from flask import Flask, request

from AsyncProcess import convert_async
from cfg_MD import save_path

app = Flask(__name__)


def make_dir(make_dir_path):
    path = make_dir_path.strip()
    if not os.path.exists(path):
        os.makedirs(path)
    return path


@app.route('/test')
def img_start():
    return json.dumps({"system": 0}, ensure_ascii=False)


@app.route('/imr-ai-service/motion_detection/convert', methods=['POST'])
def convert():
    if request.method == "POST":
        time_take = time.time()
        c_da = request.data
        data = json.loads(c_da.decode())
        file_id = data['ceph_id']
        trance_id = data['trance_log_id']

        if trance_id != 'LOCAL_USAGE':
            t_conv = threading.Thread(target=convert_async, args=(file_id, trance_id))
            t_conv.start()
            result = None
        else:
            result = convert_async(file_id=file_id, trance_log_id=trance_id)

        time_take = time.time() - time_take
        ret = not (result == -1)
        msg = {True: '成功', False: '失败'}
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
    pid = os.getpid()
    print('pid is:', pid)
    with open(os.path.join(save_path, 'app_pid.txt'), 'w') as f:
        f.writelines([str(pid)])
    app.run(
        host="0.0.0.0",
        port=int("20292"),
        debug=False, threaded=True)
