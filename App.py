from flask import Flask
from MotionDet.App_Motion import convert
from FaceRec.App_Faces import check as check_fr
from PedestrianDet.App_Pedestrian import check as check_pd
from FaceRec.App_Faces import recognize, locate, add, delete, query, reload, camera, camera2

app = Flask(__name__)


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
