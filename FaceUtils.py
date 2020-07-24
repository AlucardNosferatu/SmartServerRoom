from __future__ import absolute_import
from __future__ import print_function

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

from Configs import batch_size, epochs, new_epochs, number_of_tested_items
from Data import get_data, load_4_faces
from Networks import get_model, compute_accuracy


def train(x_train, y_train, x_test, y_test):
    input_shape, tr_pairs, tr_y, te_pairs, te_y, _, _, _, _ = get_data(
        x_train,
        y_train,
        x_test,
        y_test
    )
    model, base_network, without_dense = get_model(input_shape)
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
    input_shape, _, _, _, _, y_train_one_hot, y_test_one_hot, x_train, x_test = get_data(
        x_train,
        y_train,
        x_test,
        y_test
    )
    # for i in range(y_train_one_hot.shape[0]):
    #     cv2.imshow(str(y_train_one_hot[i]), x_train[i])
    #     cv2.waitKey()
    model, base_network, without_dense = get_model(input_shape)
    assert base_network.input_shape[1:] == input_shape
    if os.path.exists(path='Models/Siamese.h5'):
        model.load_weights(filepath='Models/Siamese.h5')
    if os.path.exists(path='Models/Softmax.h5'):
        base_network.load_weights(filepath='Models/Softmax.h5')
    if os.path.exists(path='Models/Conv.h5'):
        without_dense.load_weights(filepath='Models/Conv.h5')
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
    input_shape, tr_pairs, tr_y, te_pairs, te_y, y_train_one_hot, y_test_one_hot, x_train, x_test = get_data(
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
    # if os.path.exists(path='Models/Siamese.h5'):
    #     model.load_weights(filepath='Models/Siamese.h5')
    # if os.path.exists(path='Models/Softmax.h5'):
    #     base_network.load_weights(filepath='Models/Softmax.h5')
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
    model.save_weights(filepath='Models/Siamese.h5')
    base_network.save_weights(filepath='Models/Softmax.h5')
    without_dense.save_weights(filepath='Models/Conv.h5')


def test(x_train, y_train, x_test, y_test, extended_num_classes=None):
    print('Test start...')
    input_shape, tr_pairs, tr_y, te_pairs, te_y, _, _, _, _ = get_data(
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
    print('Model loaded.')
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
        title = np.argmax(y_pred[item])
        plt.title(str(title), loc='center')
    plt.show()


def full_process(test_num_classes=None):
    tr, te = load_4_faces()
    xtr, ytr = tr
    xte, yte = te
    if test_num_classes is None:
        train(xtr, ytr, xte, yte)
        print('Siamese training completed.')
        train_classification(xtr, ytr, xte, yte)
        print('Softmax training completed.')
    elif type(test_num_classes) is int:
        train_increment(xtr, ytr, xte, yte, test_num_classes)
    test(xtr, ytr, xte, yte, test_num_classes)


if __name__ == '__main__':
    full_process()
