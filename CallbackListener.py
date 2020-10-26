import os
import time
import json
import datetime
from flask import Flask, request

app = Flask(__name__)


@app.route('/imr-face-server/callback_listener', methods=['POST'])
def convert():
    if request.method == "POST":
        time_take = time.time()
        c_da = request.data
        print('============================')
        print(str(datetime.datetime.now()))
        try:
            data = json.loads(c_da.decode())
            for key in data:
                print('键：', key, '值：', data[key])
            ret = True
        except Exception as e:
            print(repr(e))
            ret = False
        print(str(datetime.datetime.now()))
        print('============================')
        time_take = time.time() - time_take
        msg = {True: '成功', False: '失败'}
        return json.dumps(
            {
                'msg': msg[ret],
                'timeTake': round(time_take, 4)
            },
            ensure_ascii=False
        )


if __name__ == '__main__':
    pid = os.getpid()
    print('pid is:', pid)
    with open('CBL_pid.txt', 'w') as f:
        f.writelines([str(pid)])
    app.run(
        host="0.0.0.0",
        port=int("20295"),
        debug=False, threaded=True)
