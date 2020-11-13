import datetime
import os

import cv2
import insightface
import numpy as np

from cfg_FR import face_folder_path
from logger_FR import logger

model = insightface.app.FaceAnalysis()
ctx_id = -1
model.prepare(ctx_id=ctx_id, nms=0.4)

vector_list = []
name_list = []


def reload_records():
    global vector_list, name_list
    prev = len(name_list)
    vector_list.clear()
    name_list.clear()
    face_files = os.listdir(face_folder_path)
    print(face_files)
    logger.debug(str(face_files))
    for file in face_files:
        valid = file.endswith('.png')
        valid = valid or file.endswith('.jpg')
        valid = valid or file.endswith('.PNG')
        valid = valid or file.endswith('.JPG')
        if valid:
            name_list.append(file)
            img_path = os.path.join(face_folder_path, file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (int(img.shape[1] / 4), int(img.shape[0] / 4)))
            faces = model.get(img)
            face = faces[0]
            vector = face.embedding / face.embedding_norm
            vector_list.append(vector)
    print(str(list(set(face_files).difference(set(name_list)))))
    logger.debug(str(list(set(face_files).difference(set(name_list)))))
    now = len(name_list)
    return now, now - prev


reload_records()


def test_recognizer(img_array):
    now = datetime.datetime.now()
    img_array = cv2.resize(img_array, (int(img_array.shape[1] / 4), int(img_array.shape[0] / 4)))
    faces = model.get(img_array)
    area_list = []
    for face in faces:
        bbox = face.bbox
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]
        dx = x2 - x1
        dy = y2 - y1
        area = dx * dy
        area_list.append(area)
    max_area = max(area_list)
    index = area_list.index(max_area)
    face = faces[index]
    vector = face.embedding / face.embedding_norm
    dist_list = []
    for index, v in enumerate(vector_list):
        dist = np.linalg.norm(vector - v)
        dist_list.append(dist)
    then = datetime.datetime.now()
    print(str(then - now))
    min_dist = float(min(dist_list))
    most_similar = dist_list.index(min_dist)
    name = name_list[most_similar]
    return name, min_dist


if __name__ == '__main__':
    img = cv2.imread('Samples/test.jpg')
    test_recognizer(img)
