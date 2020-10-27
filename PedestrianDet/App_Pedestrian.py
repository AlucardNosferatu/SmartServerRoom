import json
import logging
import os
import threading
import time

from flask import Flask, request

from cfg_PD import save_path
from utils_PD import download, detect_body_parts

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
        c_da = request.data
        data = json.loads(c_da.decode())
        recode_id = data['recodeId']
        equipment_id = data['equipmentId']
        time_take = time.time()
        print('下载文件', file_id)
        file_name = download(req_id=file_id, from_temp=True)
        if not (file_name.endswith('.jpg') or file_name.endswith('.png')):
            result = file_name
            ret = False
        else:
            t_ped = threading.Thread(
                target=detect_body_parts,
                args=(
                    file_id,
                    file_name,
                    recode_id,
                    equipment_id
                )
            )
            t_ped.start()
            result = '异步处理（回调）模式'
            ret = True
        time_take = time.time() - time_take
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
        port=int("20294"),
        debug=False, threaded=True)
