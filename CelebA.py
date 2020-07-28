import os
import random
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

from Configs import test_ratio, limiter, limiter_test

num_classes_celeb_a = 100


def get_celeb_a():
    path = "C:/BaiduNetdiskDownload/15_Ce.leb.Faces Att.ribu.tes Da.ta.set (CelebA)/Anno/identity_CelebA.txt"
    img_path = "C:/BaiduNetdiskDownload/15_Ce.leb.Faces Att.ribu.tes Da.ta.set (CelebA)/Img/img_align_celeba"
    f = open(path, mode='r', encoding='utf-8')
    lines = f.readlines()
    id_dict = {}
    for line in lines:
        temp = line.strip().split(' ')
        if int(temp[1]) > num_classes_celeb_a:
            continue
        id_dict[temp[1]] = id_dict.get(temp[1], []) + [temp[0]]
    img_list = []
    label_list = []
    for key in id_dict:
        label = int(key) - 1
        file_list = id_dict[key]
        temp = file_list.copy()
        while len(file_list) < 40:
            file_list.append(random.choice(temp))
        for file in file_list:
            label_list.append(label)
            img = cv2.imread(os.path.join(img_path, file))
            img = cv2.resize(img, (224, 224))
            img_list.append(img)
    x = np.array(img_list)
    y = np.array(label_list)
    state = np.random.get_state()
    np.random.shuffle(x)
    np.random.set_state(state)
    np.random.shuffle(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_ratio)

    train_gen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        cval=0,
        channel_shift_range=0,
        vertical_flip=False
    )
    train_iter = train_gen.flow(x_train, y_train, batch_size=1)
    x_train_aug = []
    y_train_aug = []
    if limiter > x_train.shape[0]:
        bound = limiter - x_train.shape[0]
        for i in tqdm(range(bound)):
            x_batch, y_batch = train_iter.next()
            x_train_aug.append(np.squeeze(x_batch))
            y_train_aug.append(np.squeeze(y_batch))
        x_train = np.concatenate([x_train, np.array(x_train_aug)], axis=0)
        y_train = np.concatenate([y_train, np.array(y_train_aug)], axis=0)
    else:
        bound = limiter
        x_train = x_train[:bound]
        y_train = y_train[:bound]

    test_gen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        cval=0,
        channel_shift_range=0,
        vertical_flip=False
    )
    test_iter = test_gen.flow(x_test, y_test, batch_size=1)
    x_test_aug = []
    y_test_aug = []
    if limiter_test > x_test.shape[0]:
        bound = limiter_test - x_test.shape[0]
        for i in tqdm(range(bound)):
            x_batch, y_batch = test_iter.next()
            x_test_aug.append(np.squeeze(x_batch))
            y_test_aug.append(np.squeeze(y_batch))
        x_test = np.concatenate([x_test, np.array(x_test_aug)], axis=0)
        y_test = np.concatenate([y_test, np.array(y_test_aug)], axis=0)
    else:
        bound = limiter_test
        x_test = x_test[:bound]
        y_test = y_test[:bound]
    return [x_train, y_train], [x_test, y_test]


if __name__ == '__main__':
    get_celeb_a()
