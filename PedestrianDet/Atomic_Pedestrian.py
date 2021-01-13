import json
import os
import time
from flask import Flask, request

from cfg_PD import save_path
from utils_PD import b64string2array
# from Haar import test_on_array
from UseImageAI import test_on_array

app = Flask(__name__)


@app.route('/imr-ai-service/atomic_functions/detect_pedestrian', methods=['POST'])
def detect_p():
    if request.method == "POST":
        c_da = request.data
        data = json.loads(c_da.decode())
        img_string = data['imgString']
        img = b64string2array(img_string)
        time_take = time.time()
        if img is None:
            result = []
        else:
            print('开始')
            result = test_on_array(img)
            print('结束')
        time_take = time.time() - time_take
        # os.remove(path)
        return json.dumps({'res': result, 'timeTake': round(time_take, 4)},
                          ensure_ascii=False)


if __name__ == '__main__':
    pid = os.getpid()
    print('pid is:', pid)
    with open(os.path.join(save_path, 'atomic_pid.txt'), 'w') as f:
        f.writelines([str(pid)])
    app.run(
        host="0.0.0.0",
        port=int("12245"),
        debug=False, threaded=True)
