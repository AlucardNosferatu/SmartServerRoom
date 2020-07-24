import os
import random

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

from Configs import data_subsample, img_size, test_ratio, num_classes, limiter


def create_pairs(x, digit_indices, extended_num_classes=None):
    if extended_num_classes is None:
        extended_num_classes = num_classes
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(extended_num_classes)]) - 1
    for d in range(extended_num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, extended_num_classes)
            dn = (d + inc) % extended_num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


def load_4_faces():
    img_array_list = []
    label_list = []
    path = "C:\\Users\\16413\\Documents\\GitHub\\YOLO\\faces\\Faces"
    dir_list = os.listdir(path)
    for directory in dir_list:
        assert directory.isnumeric()
        label = int(directory)
        directory = os.path.join(path, directory)
        img_list = os.listdir(directory)
        for img in tqdm(img_list[:data_subsample]):
            img = os.path.join(directory, img)
            img_array = cv2.imread(img)
            img_array = cv2.resize(img_array, (img_size, img_size))
            img_array_list.append(img_array)
            label_list.append(label)
    x = np.array(img_array_list)
    y = np.array(label_list)

    state = np.random.get_state()
    np.random.shuffle(x)
    np.random.set_state(state)
    np.random.shuffle(y)
    print(x.shape)
    # for i in range(x.shape[0]):
    #     cv2.imshow(str(y[i]), x[i])
    #     cv2.waitKey()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_ratio)
    datagen = ImageDataGenerator(
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
    data_iter = datagen.flow(x_train, y_train, batch_size=1)
    x_train_aug = []
    y_train_aug = []
    for i in tqdm(range(limiter)):
        x_batch, y_batch = data_iter.next()
        x_train_aug.append(np.squeeze(x_batch))
        y_train_aug.append(np.squeeze(y_batch))
    x_train = np.array(x_train_aug)
    y_train = np.array(y_train_aug)

    return (x_train, y_train), (x_test, y_test)


def get_data(x_train, y_train, x_test, y_test, extended_num_classes=None):
    input_shape = x_train.shape[1:]
    assert input_shape[0:2] == (img_size, img_size)
    # create training+test positive and negative pairs
    if extended_num_classes is None:
        y_train_one_hot = to_categorical(y_train, num_classes=num_classes)
        y_test_one_hot = to_categorical(y_test, num_classes=num_classes)

        digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]
        tr_pairs, tr_y = create_pairs(x_train, digit_indices)
        digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
        te_pairs, te_y = create_pairs(x_test, digit_indices)
    else:
        assert type(extended_num_classes) is int
        y_train_one_hot = to_categorical(y_train, num_classes=extended_num_classes)
        y_test_one_hot = to_categorical(y_test, num_classes=extended_num_classes)

        digit_indices = [np.where(y_train == i)[0] for i in range(extended_num_classes)]
        tr_pairs, tr_y = create_pairs(x_train, digit_indices, extended_num_classes)
        digit_indices = [np.where(y_test == i)[0] for i in range(extended_num_classes)]
        te_pairs, te_y = create_pairs(x_test, digit_indices, extended_num_classes)

    return input_shape, tr_pairs, tr_y, te_pairs, te_y, y_train_one_hot, y_test_one_hot, x_train, x_test
