import os
import dlib
import numpy
import cv2

predictor_path = 'Models/shape_predictor_68_face_landmarks.dat'
face_rc_model_path = 'Models/dlib_face_recognition_resnet_model_v1.dat'
# face_folder_path = 'C:/Users/16413/Documents/GitHub/YOLO/faces/Faces/forDlib'
face_folder_path = 'Backup/Faces'
test_img_path = "Samples/test.jpg"


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


def init_detectors():
    # 人脸检测器
    detector = dlib.get_frontal_face_detector()

    # 关键点检测器
    feature_point = dlib.shape_predictor(predictor_path)

    # 人脸参数模型
    feature_model = dlib.face_recognition_model_v1(face_rc_model_path)
    return detector, feature_point, feature_model


def get_recorded_features():
    detector, feature_point, feature_model = init_detectors()
    # 候选人特征向量列表
    descriptors = []
    name_list, img_list = read_data(face_folder_path)
    for i in img_list:
        # 人脸检测
        dets = detector(i, 1)

        for k, d in enumerate(dets):
            cv2.imshow('detected face:', i[d.top():d.bottom(), d.left():d.right(), :])
            cv2.waitKey()
            # 关键点检测 68点
            shape = feature_point(i, d)
            key_points = list(shape.parts())
            i_copy = i.copy()
            for point in key_points:
                i_copy = cv2.circle(i_copy, (point.x, point.y), 1, (255, 0, 0), 4)
            cv2.imshow('detected face:', i_copy)
            cv2.waitKey()
            # 提取特征，128维
            face_feature = feature_model.compute_face_descriptor(i, shape)

            v = numpy.array(face_feature)
            descriptors.append(v)
    return descriptors, name_list


def test_single_image():
    detector, feature_point, feature_model = init_detectors()
    descriptors, name_list = get_recorded_features()
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


def test_cam():
    detector, feature_point, feature_model = init_detectors()
    descriptors, name_list = get_recorded_features()
    '''
    对单张人脸进行识别
    '''
    # sample = cv2.VideoCapture(0 +
    file_path = "Outputs/010tMonitorCollect202008030345203367831f74d7_281_0.mp4"
    sample = cv2.VideoCapture(file_path)
    while sample.isOpened():
        ret, test_img = sample.read()
        if test_img is not None:
            dets = detector(test_img, 1)
            test_feature = None
            cv2.imshow('detected face:', test_img)
            for k, d in enumerate(dets):
                shape = feature_point(test_img, d)
                key_points = list(shape.parts())
                test_img_copy = test_img.copy()
                for point in key_points:
                    test_img_copy = cv2.circle(test_img_copy, (point.x, point.y), 1, (255, 0, 0), 4)
                cv2.imshow('detected face:', test_img_copy)
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
        k = cv2.waitKey(50)
        if k & 0xff == ord('q'):
            break
    sample.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test_cam()
