import os

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, Flatten

batch_size = 32
img_dir_path = 'C:/BaiduNetdiskDownload/202005/202005.v2'


def build_model():
    vgg = VGG16(include_top=False, weights='imagenet')
    x = Input(shape=(224, 224, 3))
    y = vgg(x)
    y = Flatten()(y)
    y = Dense(2, activation='softmax')(y)
    new_vgg = Model(x, y)
    return new_vgg


def data_generator():
    while True:
        with open('../valid_file.txt', mode='r', encoding='utf-8') as f:
            labels = []
            images = []
            for i in range(batch_size):
                line = f.readline()
                if line:
                    img_name, label = line.strip().split('\t')
                    labels.append(labels)
                    img_path = os.path.join(img_dir_path, img_name)
                    image = cv2.imread(img_path)
                    image = cv2.resize(image, (224, 224))
                    images.append(image)
            if len(images) == batch_size:
                x = np.array(images)
                y = np.array(labels)
                y = to_categorical(y, num_classes=2)
                yield x, y


def train_model():
    model = build_model()
    model.fit_generator(generator=data_generator(), epochs=100)
