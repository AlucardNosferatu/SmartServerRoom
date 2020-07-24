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

num_classes = 3
epochs = 1000
new_epochs = 100
number_of_tested_items = 25
img_size = 224
batch_size = 8
limiter = 500


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
    for directory in dir_list:
        assert directory.isnumeric()
        label = int(directory)
        directory = os.path.join(path, directory)
        img_list = os.listdir(directory)
        for img in tqdm(img_list):
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

    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

    datagen.fit(x)
    data_iter = datagen.flow(x, y, batch_size=1)
    x_train_aug = []
    y_train_aug = []
    for i in tqdm(range(1000)):
        x_batch, y_batch = data_iter.next()
        x_train_aug.append(np.squeeze(x_batch))
        y_train_aug.append(np.squeeze(y_batch))
    x = np.array(x_train_aug)
    y = np.array(y_train_aug)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    print('Done')
    return (x_train[:limiter], y_train[:limiter]), (x_test, y_test)


def get_data(x_train, y_train, x_test, y_test, extended_num_classes=None):
    x_train = x_train.reshape(x_train.shape[0], img_size, img_size, -1)
    x_test = x_test.reshape(x_test.shape[0], img_size, img_size, -1)
    print(x_train.shape)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

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

    return input_shape, tr_pairs, tr_y, te_pairs, te_y, y_train_one_hot, y_test_one_hot


def create_base_net(input_shape, extended_num_classes=None):
    input_layer = Input(shape=input_shape)
    x = Conv2D(8, (5, 5), activation='relu')(input_layer)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Conv2D(16, (5, 5), activation='tanh')(x)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    without_dense = Model(input_layer, x)
    if extended_num_classes is not None:
        assert type(extended_num_classes) is int
        output = Dense(extended_num_classes, activation='softmax')(x)
    else:
        output = Dense(num_classes, activation='softmax')(x)
    model = Model(input_layer, output)
    rms = RMSprop(lr=0.0001)
    model.compile(
        optimizer=rms,
        loss='categorical_crossentropy'
    )
    return model, without_dense


def get_model(input_shape, extended_num_classes=None):
    # network definition
    if extended_num_classes is not None:
        assert type(extended_num_classes) is int
    base_network, without_dense = create_base_net(input_shape, extended_num_classes)

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
    rms = RMSprop(lr=0.0001)
    model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])

    return model, base_network, without_dense


def test(x_train, y_train, x_test, y_test, extended_num_classes=None):
    input_shape, tr_pairs, tr_y, te_pairs, te_y, y_train_one_hot, y_test_one_hot = get_data(
        x_train,
        y_train,
        x_test,
        y_test,
        extended_num_classes
    )
    model, base_network, without_dense = get_model(
        input_shape,
        extended_num_classes
    )
    if os.path.exists(path='Models/Conv.h5'):
        without_dense.load_weights(filepath='Models/Conv.h5')
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

    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
    plt.figure(figsize=(10, 5))
    for item in range(number_of_tested_items):
        display = plt.subplot(int(number_of_tested_items / 5), 5, item + 1)
        # im = tf.keras.preprocessing.image.array_to_img(tr_pairs[item, 0], data_format=None, scale=True, dtype=None)
        # plt.imshow(im, cmap="gray")
        # display.get_xaxis().set_visible(False)
        # display.get_yaxis().set_visible(False)
        # display = plt.subplot(2, number_of_items, item + 1 + number_of_items)
        im = tf.keras.preprocessing.image.array_to_img(tr_pairs[item, 1], data_format=None, scale=True, dtype=None)
        plt.imshow(im)
        display.get_xaxis().set_visible(False)
        display.get_yaxis().set_visible(False)
        # plt.title(str(np.round(y_pred[item]).T[0].tolist()), loc='center')
        title = np.argmax(y_pred[item])
        plt.title(str(title), loc='center')
        # print(y_pred[item])
    plt.show()


def train(x_train, y_train, x_test, y_test):
    input_shape, tr_pairs, tr_y, te_pairs, te_y, _, _ = get_data(
        x_train,
        y_train,
        x_test,
        y_test
    )
    model, base_network, without_dense = get_model(input_shape)
    if os.path.exists(path='Models/Siamese.h5'):
        model.load_weights(filepath='Models/Siamese.h5')
    if os.path.exists(path='Models/Softmax.h5'):
        base_network.load_weights(filepath='Models/Softmax.h5')
    if os.path.exists(path='Models/Conv.h5'):
        without_dense.load_weights(filepath='Models/Conv.h5')
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
    model.save_weights(filepath='Models/Siamese.h5')
    base_network.save_weights(filepath='Models/Softmax.h5')
    without_dense.save_weights(filepath='Models/Conv.h5')


def train_classification(x_train, y_train, x_test, y_test):
    input_shape, _, _, _, _, y_train_one_hot, y_test_one_hot = get_data(
        x_train,
        y_train,
        x_test,
        y_test
    )
    model, base_network, without_dense = get_model(input_shape)
    if os.path.exists(path='Models/Siamese.h5'):
        model.load_weights(filepath='Models/Siamese.h5')
    if os.path.exists(path='Models/Softmax.h5'):
        base_network.load_weights(filepath='Models/Softmax.h5')
    if os.path.exists(path='Models/Conv.h5'):
        without_dense.load_weights(filepath='Models/Conv.h5')
    assert base_network.input_shape[1:] == input_shape
    tf.keras.utils.plot_model(
        model=base_network,
        to_file='base_model.png',
        show_shapes=True
    )
    cp_checkpoint = ModelCheckpoint(
        filepath='Models/Softmax.h5',
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
            y_train_one_hot,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test_one_hot),
            callbacks=[
                cp_checkpoint,
                es_checkpoint,
                tb_checkpoint
            ]
        )
    model.save_weights(filepath='Models/Siamese.h5')
    base_network.save_weights(filepath='Models/Softmax.h5')
    without_dense.save_weights(filepath='Models/Conv.h5')


def train_increment(x_train, y_train, x_test, y_test, extended_num_classes):
    input_shape, tr_pairs, tr_y, te_pairs, te_y, y_train_one_hot, y_test_one_hot = get_data(
        x_train,
        y_train,
        x_test,
        y_test,
        extended_num_classes
    )
    model, base_network, without_dense = get_model(
        input_shape,
        extended_num_classes
    )
    if os.path.exists(path='Models/Conv.h5'):
        without_dense.load_weights(filepath='Models/Conv.h5')
    es_checkpoint = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=10,
        verbose=1,
        mode='auto'
    )
    model.fit(
        [tr_pairs[:, 0], tr_pairs[:, 1]],
        tr_y,
        batch_size=batch_size,
        epochs=new_epochs,
        validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y),
        callbacks=[es_checkpoint]
    )
    base_network.fit(
        x_train,
        y_train_one_hot,
        batch_size=batch_size,
        epochs=new_epochs,
        validation_data=(x_test, y_test_one_hot),
        callbacks=[es_checkpoint]
    )
    # model.save_weights(filepath='Models/Siamese.h5')
    # base_network.save_weights(filepath='Models/Softmax.h5')
    without_dense.save_weights(filepath='Models/Conv.h5')


def full_process(mode='init'):
    tr, te = load_4_faces()
    xtr, ytr = tr
    xte, yte = te
    test_num_classes = None
    if mode == 'init':
        train(xtr, ytr, xte, yte)
        print('Siamese training completed.')
        train_classification(xtr, ytr, xte, yte)
        print('Softmax training completed.')
        test_num_classes = None
    elif mode == 'new':
        test_num_classes = 4
        train_increment(xtr, ytr, xte, yte, test_num_classes)
    test(xtr, ytr, xte, yte, test_num_classes)


if __name__ == '__main__':
    full_process()
