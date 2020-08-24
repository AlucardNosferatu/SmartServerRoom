from flask import Flask, request
import os
import cv2
import json
import time
from QRCode import test_on_array
import base64
from skimage import io
import numpy as np
import logging

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


@app.route('/decode', methods=['POST'])
def img_test():
    log_file_name = 'logger-' + time.strftime('%Y-%m-%d', time.localtime(time.time())) + '.log'
    log_file_str = log_file_folder + os.sep + log_file_name
    if not os.path.exists(log_file_str):
        handler = logging.FileHandler(log_file_str, encoding='UTF-8')
        handler.setFormatter(logging_format)
        app.logger.addHandler(handler)

    if request.method == "POST":
        start_time = time.time()
        c_da = request.data
        data = str(c_da, "utf-8")

        data = eval(c_da.decode())
        img_string = data['imgString'].encode()
        img_string = img_string.decode()
        # print(imgString)
        img = np.array([])
        if "base64," in str(img_string):
            img_string = data['imgString'].encode().split(b';base64,')[-1]

        if ".jpg" in str(img_string) or ".png" in str(img_string):
            # print(imgString)
            app.logger.info(data)
            img_string = img_string.replace("\n", "")
            img_rgb = io.imread(img_string)
            img = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)

        if len(img_string) > 200:
            img_string = base64.b64decode(img_string)
            nparr = np.frombuffer(img_string, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        time_take = time.time()

        result = test_on_array(img)

        time_take = time.time() - time_take

        if "fileName" in data.keys():
            app.logger.info("recognition  return:{d},use time:{t}".format(d=result, t=time_take))
            return json.dumps({data['fileName']: [{'value': result}]}, ensure_ascii=False)
        # os.remove(path)
        return json.dumps({'res': result, 'timeTake': round(time_take, 4)},
                          ensure_ascii=False)


if __name__ == '__main__':
    app.run(
        host="0.0.0.0",
        port=int("2029"),
        debug=False, threaded=True)
