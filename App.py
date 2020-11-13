import os
import path_cfg
from flask import Flask
from FaceRec.FaceRec.App_Faces import check as check_fr
from FaceRec.FaceRec.App_Faces import recognize, locate, add, delete, query, reload, camera, camera2
from MotionDet.App_Motion import convert as convert_md
from PedestrianDet.App_Pedestrian import check as check_pd

app = Flask(__name__)


@app.route('/imr-ai-service/pedestrian/check/<file_id>', methods=['POST'])
def check_pedestrian(file_id):
    result = check_pd(file_id)
    return result


@app.route('/imr-ai-service/motion_detection/convert', methods=['POST'])
def convert():
    result = convert_md()
    return result


@app.route('/imr-ai-service/face_features/check/<file_id>', methods=['POST'])
def check_face(file_id):
    result = check_fr(file_id)
    return result


@app.route('/imr-ai-service/face_features/recognize/<file_id>', methods=['POST'])
def recognize_face(file_id):
    result = recognize(file_id)
    return result


@app.route('/imr-ai-service/face_features/locate/<file_id>', methods=['POST'])
def locate_face(file_id):
    result = locate(file_id)
    return result


@app.route('/imr-ai-service/face_features/add/<file_id>', methods=['POST'])
def add_face(file_id):
    result = add(file_id)
    return result


@app.route('/imr-ai-service/face_features/delete/<file_name>', methods=['POST'])
def delete_face(file_name):
    result = delete(file_name)
    return result


@app.route('/imr-ai-service/face_features/query', methods=['POST'])
def query_face():
    result = query()
    return result


@app.route('/imr-ai-service/face_features/reload', methods=['POST'])
def reload_face():
    result = reload()
    return result


@app.route('/imr-ai-service/face_features/camera_recognize', methods=['POST'])
def camera_face():
    result = camera()
    return result


@app.route('/imr-ai-service/face_features/door_open', methods=['POST'])
def door_open():
    result = camera2()
    return result


if __name__ == '__main__':
    pid = os.getpid()
    print('pid is:', pid)
    with open('app_pid.txt', 'w') as f:
        f.writelines([str(pid)])
    app.run(
        host="0.0.0.0",
        port=int("20290"),
        debug=False, threaded=True
    )
