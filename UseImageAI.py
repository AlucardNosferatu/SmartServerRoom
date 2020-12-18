import os
import datetime

import cv2
from imageai.Detection import ObjectDetection

input_path = 'Samples/Input'
output_path = 'Samples/Output'
save_path = 'P_Temp'


def init_detector(model_path='Models/resnet50_coco_best_v2.0.1.h5'):
    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(model_path)
    detector.loadModel()
    return detector


d = init_detector()
print('Model Loaded')


def batch_test(detector=d):
    custom_objects = detector.CustomObjects(person=True, car=False)
    for file in os.listdir(input_path):
        print(file)
        try:
            current_time = datetime.datetime.now()
            detections = detector.detectCustomObjectsFromImage(
                input_image=os.path.join(input_path, file),
                output_image_path=os.path.join(output_path, file),
                custom_objects=custom_objects,
                minimum_percentage_probability=65
            )
            finish_time = datetime.datetime.now()
            print('pic size:', str(cv2.imread(os.path.join(input_path, file)).shape))
            print('file size:', str(os.path.getsize(os.path.join(input_path, file))))
            print('elapsed time:', str(finish_time - current_time))
        except Exception as e:
            print(repr(e))


def test_on_array(img, detector=d):
    custom_objects = detector.CustomObjects(person=True, car=False)
    img = cv2.resize(img, (1024, 768))
    cv2.imwrite(os.path.join(save_path, 'temp.jpg'), img)
    coordinates = []
    detections = detector.detectCustomObjectsFromImage(
        input_image=os.path.join(save_path, 'temp.jpg'),
        output_image_path=os.path.join(save_path, 'temp.jpg'),
        custom_objects=custom_objects,
        minimum_percentage_probability=65,
        thread_safe=True
    )
    for det in detections:
        coordinates.append(det['box_points'])
    print('Done')
    return coordinates


if __name__ == '__main__':
    batch_test(detector=d)
