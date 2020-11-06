import logging
import os
import time
import json
import datetime
from flask import Flask, request

work_dir = __file__
if '\\' in work_dir:
    work_dir = work_dir.split('\\')
    work_dir = work_dir[:-1]
    work_dir = '\\'.join(work_dir)
else:
    work_dir = work_dir.split('/')
    work_dir = work_dir[:-1]
    work_dir = '/'.join(work_dir)
save_path = os.path.join(work_dir, 'C_Temp')
log_path = os.path.join(save_path, 'CB.txt')
logging.basicConfig(
    level=logging.DEBUG,
    filename=log_path,
    datefmt='%Y/%m/%d %H:%M:%S',
    format='%(asctime)s - %(levelname)s - %(thread)d - %(module)s - %(funcName)s - %(lineno)d - %(message)s'
)
logger = logging.getLogger('CB')
app = Flask(__name__)


@app.route('/imr-face-server/callback_listener', methods=['POST'])
def convert():
    if request.method == "POST":
        time_take = time.time()
        c_da = request.data
        print('============================')
        logger.info('============================')
        print(str(datetime.datetime.now()))
        try:
            data = json.loads(c_da.decode())
            for key in data:
                print('键：', key, '值：', data[key])
                logger.debug('键：' + str(key) + ' ' + '值：' + str(data[key]))
            ret = True
        except Exception as e:
            print(repr(e))
            logger.error(repr(e))
            ret = False
        print(str(datetime.datetime.now()))
        print('============================')
        logger.info('============================')
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
    with open(os.path.join(save_path, 'atomic_pid.txt'), 'w') as f:
        f.writelines([str(pid)])
    app.run(
        host="0.0.0.0",
        port=int("20295"),
        debug=False, threaded=True)
