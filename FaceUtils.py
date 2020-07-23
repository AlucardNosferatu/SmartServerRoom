from __future__ import absolute_import
from __future__ import print_function

import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.layers import Conv2D, AveragePooling2D
from tensorflow.keras.layers import Input, Flatten, Dense, Lambda, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

num_classes = 4
epochs = 1000
number_of_tested_items = 20
img_size = 224
batch_size = 8


def euclid_dis(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1[0], 1


def contrastive_loss(y_true, y_pred):
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def create_pairs(x, digit_indices):
    pairs = []
    labels = []

    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1

    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def load_4_faces():
    img_array_list = []
    label_list = []
    path = "C:\\Users\\16413\\Documents\\GitHub\\YOLO\\faces\\Faces"
    dir_list = os.listdir(path)
    label = 0
    for directory in dir_list:
        directory = os.path.join(path, directory)
        img_list = os.listdir(directory)
        for img in tqdm(img_list):
            img = os.path.join(directory, img)
            img_array = cv2.imread(img)
            img_array = cv2.resize(img_array, (img_size, img_size))
            img_array_list.append(img_array)
            label_list.append(label)
        label += 1
    x = np.array(img_array_list)
    y = np.array(label_list)
    state = np.random.get_state()
    np.random.shuffle(x)
    np.random.set_state(state)
    np.random.shuffle(y)
    # for i in range(x.shape[0]):
    #     cv2.imshow(str(y[i]), x[i])
    #     cv2.waitKey()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    print('Done')
    return (x_train, y_train), (x_test, y_test)


def get_data():
    # the data, split between train and test sets
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    (x_train, y_train), (x_test, y_test) = load_4_faces()
    # plt.figure(0)
    # for i in range(x_train.shape[0]):
    #     plt.imshow(x_train[i, :, :])
    #     plt.title(str(y_train[i]))
    #     plt.show()
    x_train = x_train.reshape(x_train.shape[0], img_size, img_size, -1)
    x_test = x_test.reshape(x_test.shape[0], img_size, img_size, -1)
    # input_shape = (1, 28, 28)
    print(x_train.shape)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    input_shape = x_train.shape[1:]
    assert input_shape[0:2] == (img_size, img_size)

    # create training+test positive and negative pairs
    digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]
    tr_pairs, tr_y = create_pairs(x_train, digit_indices)

    digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
    te_pairs, te_y = create_pairs(x_test, digit_indices)
    return input_shape, tr_pairs, tr_y, te_pairs, te_y


def create_base_net(input_shape):
    input = Input(shape=input_shape)
    x = Conv2D(32, (5, 5), activation='relu')(input)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (5, 5), activation='tanh')(x)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(input, x)
    # model.summary()
    rms = RMSprop(lr=0.001)
    model.compile(
        optimizer=rms,
        loss='categorical_crossentropy'
    )
    return model


def get_model(input_shape):
    # network definition
    base_network = create_base_net(input_shape)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(
        function=euclid_dis,
        output_shape=eucl_dist_output_shape
    )(
        [processed_a, processed_b]
    )

    model = Model([input_a, input_b], distance)
    rms = RMSprop(lr=0.001)
    model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])

    return model, base_network


def train():
    input_shape, tr_pairs, tr_y, te_pairs, te_y = get_data()
    model, base_network = get_model(input_shape)
    if os.path.exists(path='Models/Siamese.h5'):
        model.load_weights(filepath='Models/Siamese.h5')
    if os.path.exists(path='Models/Base.h5'):
        base_network.load_weights(filepath='Models/Base.h5')
    assert model.input_shape[0][1:] == input_shape
    tf.keras.utils.plot_model(
        model=model,
        to_file='model.png',
        show_shapes=True
    )
    cp_checkpoint = ModelCheckpoint(
        filepath='Models/Siamese.h5',
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_freq='epoch',
        save_weights_only=True
    )
    es_checkpoint = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=10,
        verbose=1,
        mode='auto'
    )
    tb_checkpoint = TensorBoard(
        log_dir='TensorBoard',
        histogram_freq=1,
        write_images=True,
        update_freq='epoch'
    )
    with tf.device("/gpu:0"):
        model.fit(
            [tr_pairs[:, 0], tr_pairs[:, 1]],
            tr_y,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y),
            callbacks=[
                cp_checkpoint,
                es_checkpoint,
                tb_checkpoint
            ]
        )


def test():
    input_shape, tr_pairs, tr_y, te_pairs, te_y = get_data()
    model, base_network = get_model(input_shape)
    if os.path.exists(path='Models/Siamese.h5'):
        model.load_weights(filepath='Models/Siamese.h5')
    if os.path.exists(path='Models/Base.h5'):
        base_network.load_weights(filepath='Models/Base.h5')
    assert model.input_shape[0][1:] == input_shape
    tf.keras.utils.plot_model(
        model=model,
        to_file='model.png',
        show_shapes=True
    )
    y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    tr_acc = compute_accuracy(tr_y, y_pred)

    y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
    te_acc = compute_accuracy(te_y, y_pred)

    y_pred = base_network.predict(tr_pairs[:, 1])
    # y_pred_base_net = base_network.predict()

    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
    plt.figure(figsize=(10, 3))
    for item in range(number_of_tested_items):
        display = plt.subplot(1, number_of_tested_items, item + 1)
        # im = tf.keras.preprocessing.image.array_to_img(tr_pairs[item, 0], data_format=None, scale=True, dtype=None)
        # plt.imshow(im, cmap="gray")
        # display.get_xaxis().set_visible(False)
        # display.get_yaxis().set_visible(False)
        # display = plt.subplot(2, number_of_items, item + 1 + number_of_items)
        im = tf.keras.preprocessing.image.array_to_img(tr_pairs[item, 1], data_format=None, scale=True, dtype=None)
        plt.imshow(im, cmap="gray")
        display.get_xaxis().set_visible(False)
        display.get_yaxis().set_visible(False)
        # plt.title(str(np.round(y_pred[item]).T[0].tolist()), loc='center')
        title = np.argmax(y_pred[item])
        plt.title(str(title), loc='center')
        # print(y_pred[item])
    plt.show()


def train_classification():
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    (x_train, y_train), (x_test, y_test) = load_4_faces()
    # plt.figure(0)
    # for i in range(x_train.shape[0]):
    #     plt.imshow(x_train[i, :, :])
    #     plt.title(str(y_train[i]))
    #     plt.show()
    x_train = x_train.reshape(x_train.shape[0], img_size, img_size, -1)
    x_test = x_test.reshape(x_test.shape[0], img_size, img_size, -1)
    # input_shape = (1, 28, 28)
    print(x_train.shape)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)
    input_shape = x_train.shape[1:]
    assert input_shape[0:2] == (img_size, img_size)
    model, base_network = get_model(input_shape)
    if os.path.exists(path='Models/Siamese.h5'):
        model.load_weights(filepath='Models/Siamese.h5')
    if os.path.exists(path='Models/Base.h5'):
        base_network.load_weights(filepath='Models/Base.h5')
    assert base_network.input_shape[1:] == input_shape
    tf.keras.utils.plot_model(
        model=base_network,
        to_file='base_model.png',
        show_shapes=True
    )
    cp_checkpoint = ModelCheckpoint(
        filepath='Models/Base.h5',
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_freq='epoch',
        save_weights_only=True
    )
    es_checkpoint = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=10,
        verbose=1,
        mode='auto'
    )
    tb_checkpoint = TensorBoard(
        log_dir='TensorBoard',
        histogram_freq=1,
        write_images=True,
        update_freq='epoch'
    )
    with tf.device("/gpu:0"):
        base_network.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test),
            callbacks=[
                cp_checkpoint,
                es_checkpoint,
                tb_checkpoint
            ]
        )


if __name__ == '__main__':
    train()
    train_classification()
    # test()
    # load_4_faces()
