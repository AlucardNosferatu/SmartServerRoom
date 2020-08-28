import os

import cv2
import dlib
import numpy
from dlib import rectangle

predictor_path = 'Models/shape_predictor_68_face_landmarks.dat'
face_rc_model_path = 'Models/dlib_face_recognition_resnet_model_v1.dat'
face_folder_path = 'C:/Users/16413/Documents/GitHub/YOLO/faces/Faces/forDlib'
# face_folder_path = 'Backup/Faces'
test_img_path = "Samples/test.jpg"


def init_detectors():
    # 人脸检测器
    d = dlib.get_frontal_face_detector()

    # 关键点检测器
    fp_d = dlib.shape_predictor(predictor_path)

    # 人脸参数模型
    fm = dlib.face_recognition_model_v1(face_rc_model_path)
    return d, fp_d, fm


detector, feature_point, feature_model = init_detectors()


# 读取人脸集、人脸标签
def read_data(path):
    try:
        pic_name_list = os.listdir(path)
        pic_list = []
        for i in pic_name_list:
            whole_path = os.path.join(path, i)
            img = cv2.imread(whole_path)
            img = cv2.resize(img, (int(img.shape[1] / 4), int(img.shape[0] / 4)))
            pic_list.append(img)
    except IOError:
        print('read error')
        return False
    else:
        print('read successfully')
        return pic_name_list, pic_list


def get_recorded_features():
    # 候选人特征向量列表
    descriptors = []
    name_list, img_list = read_data(face_folder_path)
    for i in img_list:
        # 人脸检测
        dets = detector(i, 1)

        for k, d in enumerate(dets):
            # cv2.imshow('detected face:', i[d.top():d.bottom(), d.left():d.right(), :])
            # cv2.waitKey()
            # 关键点检测 68点
            shape = feature_point(i, d)
            key_points = list(shape.parts())
            i_copy = i.copy()
            for point in key_points:
                i_copy = cv2.circle(i_copy, (point.x, point.y), 1, (255, 0, 0), 4)
            # cv2.imshow('detected face:', i_copy)
            # cv2.waitKey()
            # 提取特征，128维
            face_feature = feature_model.compute_face_descriptor(i, shape)
            v = numpy.array(face_feature)
            descriptors.append(v)
    return descriptors, name_list


descriptors, name_list = get_recorded_features()


def test_single_image():
    '''
    对单张人脸进行识别
    '''
    test_img = cv2.imread(test_img_path)
    dets = detector(test_img, 1)
    test_feature = None
    for k, d in enumerate(dets):
        shape = feature_point(test_img, d)
        test_feature = feature_model.compute_face_descriptor(test_img, shape)
        test_feature = numpy.array(test_feature)

        dist = []
        count = 0
        for i in descriptors:
            dist_ = numpy.linalg.norm(i - test_feature)
            print('%s : %f' % (name_list[count], dist_))
            dist.append(dist_)
            count += 1

        # 返回距离最小的下标
        min_dist = numpy.argmin(dist)

        # 截取姓名字符串，去掉末尾的.jpg
        result = name_list[min_dist][:-4]
        print(result)


def test_cam(file_path="Samples/00010001689000000.mp4", file_name='00010001689000000.mp4'):
    # descriptors, name_list = get_recorded_features()
    sample = cv2.VideoCapture(file_path)
    # sample.set(cv2.CAP_PROP_POS_FRAMES, 7000)
    f_count = -1
    while sample.isOpened():
        # k = cv2.waitKey(50)
        print(sample.get(cv2.CAP_PROP_POS_FRAMES))
        ret, test_img = sample.read()
        if test_img is not None:
            # cv2.imshow('cam:', test_img)
            # if k & 0xff == ord('q'):
            #     break
            # elif k & 0xff == ord('e'):
            #     sample.grab()
            #     continue
            detected = detector(test_img, 1)
            for k, d in enumerate(detected):
                x1 = d.left()
                x2 = d.right()
                y1 = d.top()
                y2 = d.bottom()
                w = x2 - x1
                h = y2 - y1
                dw = int(0.25 * w)
                dh = int(0.25 * h)
                x1 -= dw
                if x1 < 0:
                    x1 = 0
                x2 += dw
                if x2 > test_img.shape[1] - 1:
                    x2 = test_img.shape[1] - 1
                y1 -= dh
                if y1 < 0:
                    y1 = 0
                y2 += dh
                if y2 > test_img.shape[0] - 1:
                    y2 = test_img.shape[0] - 1

                # cv2.imshow('detected face:', test_img[y1:y2, x1:x2, :])
                f_count += 1
                cv2.imwrite('Outputs/' + file_name.split('.')[0] + '_' + str(f_count) + '.jpg',
                            test_img[y1:y2, x1:x2, :])
                shape = feature_point(test_img, d)
                key_points = list(shape.parts())
                test_img_copy = test_img.copy()
                for point in key_points:
                    test_img_copy = cv2.circle(test_img_copy, (point.x, point.y), 1, (255, 0, 0), 4)
                # cv2.imshow('cam:', test_img_copy)
                cv2.imwrite('Outputs/' + file_name.split('.')[0] + '_' + str(f_count) + '_p.jpg',
                            test_img_copy[y1:y2, x1:x2, :])
                # test_feature = feature_model.compute_face_descriptor(test_img, shape)
                # test_feature = numpy.array(test_feature)
                # dist = []
                # count = 0
                # for i in descriptors:
                #     dist_ = numpy.linalg.norm(i - test_feature)
                #     print('%s : %f' % (name_list[count], dist_))
                #     dist.append(dist_)
                #     count += 1
                # # 返回距离最小的下标
                # min_dist = numpy.argmin(dist)
                # # 截取姓名字符串，去掉末尾的.jpg
                # result = name_list[min_dist][:-4]
                # print(result)
    sample.release()
    cv2.destroyAllWindows()


def test_detector(img_array):
    detected = list(detector(img_array, 1))
    detected = [
        [
            rect.tl_corner().x,
            rect.tl_corner().y,
            rect.br_corner().x,
            rect.br_corner().y
        ] for rect in detected
    ]
    return detected


def test_landmarks(img_array):
    d_rect = rectangle(0, 0, img_array.shape[1] - 1, img_array.shape[0] - 1)
    shape = feature_point(img_array, d_rect)
    key_points = list(shape.parts())
    key_points = [[point.x, point.y] for point in key_points]
    return key_points


def test_recognizer(img_array):
    d_rect = rectangle(0, 0, img_array.shape[1] - 1, img_array.shape[0] - 1)
    shape = feature_point(img_array, d_rect)
    test_feature = feature_model.compute_face_descriptor(img_array, shape)
    test_feature = numpy.array(test_feature)
    dist = []
    count = 0
    for i in descriptors:
        dist_ = numpy.linalg.norm(i - test_feature)
        print('%s : %f' % (name_list[count], dist_))
        dist.append(dist_)
        count += 1

    # 返回距离最小的下标
    min_dist = numpy.argmin(dist)

    # 截取姓名字符串，去掉末尾的.jpg
    result = name_list[min_dist][:-4]
    return result


if __name__ == '__main__':
    dir_path = '../Faces'
    file_list = os.listdir(dir_path)
    for i in file_list:
        if i.endswith('.mp4'):
            fp = os.path.join(dir_path, i)
            test_cam(file_path=fp, file_name=i)
