import datetime
import os

import cv2
import insightface
import numpy as np

from cfg_FR import face_folder_path
from logger_FR import logger
from utils_FR import array2b64string, process_request

model = insightface.app.FaceAnalysis()
ctx_id = -1
model.prepare(ctx_id=ctx_id, nms=0.4)

vector_list = []
name_list = []


def reload_records(align=False, use_dbf=False):
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
            if align:
                faces = model.get(img)
                face = faces[0]
                vector = face.embedding / face.embedding_norm
            else:
                if use_dbf:
                    b64str = array2b64string(img).decode()
                    result = process_request('fd_dbf', req_dict={'imgString': b64str})
                    bbox = result['res'][0]
                else:
                    result, _ = model.det_model.detect(img, threshold=0.8, scale=1.0)
                    bbox = result[0, 0:4]
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                face = img[y1:y2, x1:x2, :]
                face = cv2.resize(face, (112, 112))
                embedding = model.rec_model.get_embedding(face).flatten()
                embedding_norm = np.linalg.norm(embedding)
                vector = embedding / embedding_norm
            vector_list.append(vector)
    print(str(list(set(face_files).difference(set(name_list)))))
    logger.debug(str(list(set(face_files).difference(set(name_list)))))
    now = len(name_list)
    return now, now - prev


reload_records()


def test_recognizer(img_array, align=False, use_dbf=False):
    now = datetime.datetime.now()
    img_array = cv2.resize(img_array, (int(img_array.shape[1] / 4), int(img_array.shape[0] / 4)))
    if align:
        faces = model.get(img_array)
        length = len(faces)
    else:
        if use_dbf:
            b64str = array2b64string(img).decode()
            result = process_request('fd_dbf', req_dict={'imgString': b64str})
            faces = result['res']
            length = len(faces)
        else:
            faces, _ = model.det_model.detect(img, threshold=0.8, scale=1.0)
            length = faces.shape[0]
    area_list = []
    for i in range(length):
        if align:
            bbox = faces[i].bbox
        else:
            if use_dbf:
                bbox = faces[i]
            else:
                bbox = faces[i, 0:4]
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        dx = x2 - x1
        dy = y2 - y1
        area = dx * dy
        area_list.append(area)
    if len(area_list) > 0:
        max_area = max(area_list)
        index = area_list.index(max_area)
        if align:
            face = faces[index]
            vector = face.embedding / face.embedding_norm
        else:
            if use_dbf:
                bbox = faces[index]
            else:
                bbox = faces[index, 0:4]
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
            face = img[y1:y2, x1:x2, :]
            face = cv2.resize(face, (112, 112))
            embedding = model.rec_model.get_embedding(face).flatten()
            embedding_norm = np.linalg.norm(embedding)
            vector = embedding / embedding_norm
        dist_list = []
        for index, v in enumerate(vector_list):
            dist = np.linalg.norm(vector - v)
            dist_list.append(dist)
        min_dist = float(min(dist_list))
        most_similar = dist_list.index(min_dist)
        name = name_list[most_similar]
        then = datetime.datetime.now()
        print(str(then - now))
        return name, min_dist
    else:
        then = datetime.datetime.now()
        print(str(then - now))
        return []


if __name__ == '__main__':
    img = cv2.imread('Samples/test.jpg')
    result = test_recognizer(img)
    print(result)
