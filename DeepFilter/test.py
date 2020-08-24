import os

import cv2
import numpy as np
from tensorflow.python.keras.applications.vgg16 import preprocess_input
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array

from DeepFilter.train import build_model, img_dir_path

model = build_model()


def load_weight(file_path='../Models/QRCode_Detector.h5'):
    model.load_weights(filepath=file_path)


def test_model():
    file_list = os.listdir(img_dir_path)
    with open('../valid_file.txt', mode='r', encoding='utf-8') as f:
        lines = f.readlines()
    for file in file_list:
        img_path = os.path.join(img_dir_path, file)
        image = load_img(path=img_path)
        image = img_to_array(img=image)
        show_image = image.copy().astype('uint8')
        image = cv2.resize(image, (224, 224))
        image = preprocess_input(image)
        result = model.predict(np.expand_dims(image, axis=0))
        result = str(np.argmax(result[0]))
        label = file + '\t' + result + '\n'
        print(file)
        if label in lines:
            cv2.imshow(result + ' ok', show_image)
        else:
            cv2.imshow(result + ' ng', show_image)
        cv2.waitKey()


def test_on_array(img_array):
    img_array = cv2.resize(img_array, (224, 224))
    result = model.predict(np.expand_dims(img_array, axis=0))
    cls = np.argmax(result[0])
    return [str(cls), str(result[0][cls])]


if __name__ == '__main__':
    test_model()
