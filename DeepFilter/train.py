import os
import cv2
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

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
    labels = []
    images = []
    with open('../valid_file.txt', mode='r', encoding='utf-8') as f:
        lines = f.readlines()
    while True:
        while len(images) < batch_size:
            line = random.choice(lines)
            img_name, label = line.strip().split('\t')
            labels.append(int(label))
            img_path = os.path.join(img_dir_path, img_name)
            image = cv2.imread(img_path)
            image = cv2.resize(image, (224, 224))
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


def train_model():
    model = build_model()
    cp_checkpoint = ModelCheckpoint(
        filepath='Models/QRCode_Detector.h5',
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
        optimizer=tf.keras.optimizers.Adam(lr=0.001)
    )
    model.fit_generator(
        generator=data_generator(),
        steps_per_epoch=100,
        epochs=100,
        callbacks=[
            cp_checkpoint,
            es_checkpoint,
            tb_checkpoint
        ]
    )


if __name__ == '__main__':
    # data_generator()
    train_model()
