import os
import random

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical

batch_size = 32
train_steps = 800
img_dir_path = 'D:/BaiduNetdiskDownload/202005/202005.v2'


# img_dir_path = '/tmp/PR'


def build_model():
    vgg = VGG16(include_top=False, weights='imagenet')
    x = Input(shape=(224, 224, 3))
    y = vgg(x)
    y = Flatten()(y)
    y = Dense(2, activation='softmax')(y)
    new_vgg = Model(x, y)
    return new_vgg


def data_generator(mode='train'):
    labels = []
    images = []
    with open('../valid_file.txt', mode='r', encoding='utf-8') as f:
        lines = f.readlines()
        if mode == 'train':
            lines = lines[:train_steps * batch_size]
        elif mode == 'test':
            lines = lines[-4 * batch_size:]
        else:
            raise ValueError('Mode must be "train" or "test".')
    while True:
        while len(images) < batch_size:
            line = random.choice(lines)
            img_name, label = line.strip().split('\t')
            if labels.count(0) >= (batch_size / 2) and label == '0':
                continue
            labels.append(int(label))
            img_path = os.path.join(img_dir_path, img_name)
            image = load_img(path=img_path, target_size=(224, 224))
            image = img_to_array(img=image)
            image = preprocess_input(image)
            images.append(image)
        if len(images) >= batch_size:
            x = np.array(images[:batch_size])
            y = np.array(labels[:batch_size])
            y = to_categorical(y, num_classes=2)
            # print('Yield now')
            yield x, y
            labels = []
            images = []


def data_processor(mode='train'):
    labels = []
    images = []
    with open('../valid_file.txt', mode='r', encoding='utf-8') as f:
        lines = f.readlines()
        if mode == 'train':
            lines = lines[:train_steps * batch_size]
        elif mode == 'test':
            lines = lines[-4 * batch_size:]
        else:
            raise ValueError('Mode must be "train" or "test".')
    for line in lines:
        img_name, label = line.strip().split('\t')
        labels.append(int(label))
        img_path = os.path.join(img_dir_path, img_name)
        image = load_img(path=img_path, target_size=(224, 224))
        image = img_to_array(img=image)
        image = preprocess_input(image)
        images.append(image)
    x = np.array(images)
    y = np.array(labels)
    y = to_categorical(y, num_classes=2)
    return x, y


def train_model(use_generator=True):
    model = build_model()
    cp_checkpoint = ModelCheckpoint(
        filepath='../Models/QRCode_Detector.h5',
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_freq='epoch',
        save_weights_only=True
    )
    print("cp_checkpoint added.")
    es_checkpoint = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=5,
        verbose=1,
        mode='auto'
    )
    print("es_checkpoint added.")
    tb_checkpoint = TensorBoard(
        log_dir='../TensorBoard',
        histogram_freq=1,
        write_images=True,
        update_freq='epoch'
    )
    print("tb_checkpoint added.")
    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adam(lr=0.0001),
        metrics=['acc']
    )
    if os.path.exists('../Models/QRCode_Detector.h5'):
        model.load_weights(filepath='../Models/QRCode_Detector.h5')

    if not use_generator:
        x_train, y_train = data_processor(mode='train')
        print('train data loaded.')
        x_test, y_test = data_processor(mode='test')
        print('test data loaded.')
    else:
        x_train, y_train = np.zeros((1, 224, 224, 3)), np.zeros((1,))
        x_test, y_test = np.zeros((1, 224, 224, 3)), np.zeros((1,))
    with tf.device("/gpu:0"):
        if use_generator:
            model.fit_generator(
                generator=data_generator(mode='train'),
                validation_data=data_generator(mode='test'),
                steps_per_epoch=train_steps,
                validation_steps=2,
                epochs=100,
                callbacks=[
                    cp_checkpoint,
                    es_checkpoint,
                    tb_checkpoint
                ]
            )
        else:
            model.fit(
                x=x_train,
                y=y_train,
                validation_data=[x_test, y_test],
                batch_size=batch_size,
                steps_per_epoch=train_steps,
                validation_steps=2,
                epochs=100,
                callbacks=[
                    cp_checkpoint,
                    es_checkpoint
                ]
            )


def test_model():
    model = build_model()
    model.load_weights(filepath='../Models/QRCode_Detector.h5')
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


if __name__ == '__main__':
    # data_generator()
    # train_model(use_generator=False)
    test_model()
