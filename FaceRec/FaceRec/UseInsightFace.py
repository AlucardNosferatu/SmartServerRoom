import datetime
import os

import cv2
import insightface
import numpy as np

from cfg_FR import face_folder_path
from logger_FR import logger
from utils_FR import array2b64string, process_request

ind_rec_model = False
model = insightface.app.FaceAnalysis()
ctx_id = -1
model.prepare(ctx_id=ctx_id, nms=0.4)
if ind_rec_model:
    model2 = insightface.model_zoo.get_model('arcface_r100_v1')
    model2.prepare(ctx_id=ctx_id)

img_list = []
vector_list = []
name_list = []



def reload_records(align=True, use_dbf=True):
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
                if ind_rec_model:
                    img_l = cv2.resize(img, (112, 112))
                    img_list.append(img_l)
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


def test_detector(img_array):
    out_d = []

    # img_array = cv2.resize(img_array, (int(img_array.shape[1] / 2), int(img_array.shape[0] / 2)))
    detected = model.get(img_array)

    for d in detected:
        x1 = d.bbox[0]
        x2 = d.bbox[2]
        y1 = d.bbox[1]
        y2 = d.bbox[3]
        w = x2 - x1
        h = y2 - y1
        dw = 0.25 * w
        dh = 0.25 * h
        x1 -= dw
        if x1 < 0:
            x1 = 0
        x2 += dw
        if x2 > img_array.shape[1] - 1:
            x2 = img_array.shape[1] - 1
        y1 -= dh
        if y1 < 0:
            y1 = 0
        y2 += dh
        if y2 > img_array.shape[0] - 1:
            y2 = img_array.shape[0] - 1
        out_d.append([int(x1) * 2, int(y1) * 2, int(x2) * 2, int(y2) * 2])
    return out_d


def test_recognizer(img_array, align=True, use_dbf=True):

    now = datetime.datetime.now()
    # img_array = cv2.resize(img_array, (int(img_array.shape[1] / 2), int(img_array.shape[0] / 2)))
    if align:
        # end to end process including detection
        faces = model.get(img_array)
        length = len(faces)
    else:
        if use_dbf:
            # detect face using DBFace
            b64str = array2b64string(img).decode()
            result = process_request('fd_dbf', req_dict={'imgString': b64str})
            faces = result['res']
            length = len(faces)
        else:
            # use RetinaNet embedded in insightface
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
        # get the face with largest area
        max_area = max(area_list)
        index = area_list.index(max_area)
        if ind_rec_model:
            face = faces[index]
            x1 = int(face.bbox[0])
            y1 = int(face.bbox[1])
            x2 = int(face.bbox[2])
            y2 = int(face.bbox[3])
            img_f = img_array[y1:y2, x1:x2, :]
            img_f = cv2.resize(img_f, (112, 112))
            sim_list = []
            for index, img_l in enumerate(img_list):
                sim = model2.compute_sim(img_f, img_l)
                sim_list.append(sim)
            min_dist = float(max(sim_list))
            most_similar = sim_list.index(min_dist)
            name = name_list[most_similar]
        else:
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
                # cv2.imshow('detected largest', face)
                # cv2.waitKey()
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
        return 'no_aligned_faces_detected', 99.0


if __name__ == '__main__':
    path = 'Samples'
    file_list = os.listdir(path)
    for file in file_list:
        img = cv2.imread(os.path.join('Samples', file))
        result = test_recognizer(img)
        print(file, result)
