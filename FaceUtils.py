from __future__ import absolute_import
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
import random
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.layers import Conv2D, Activation, AveragePooling2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint

num_classes = 10
epochs = 20


def euclid_dis(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


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


def create_base_net(input_shape):
    input = Input(shape=input_shape)
    x = Conv2D(4, (5, 5), activation='tanh')(input)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Conv2D(16, (5, 5), activation='tanh')(x)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(10, activation='tanh')(x)
    model = Model(input, x)
    model.summary()

    return model


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def get_data():
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    # input_shape = (1, 28, 28)
    print(x_train.shape)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    input_shape = x_train.shape[1:]
    assert input_shape == (28, 28, 1)

    # create training+test positive and negative pairs
    digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]
    tr_pairs, tr_y = create_pairs(x_train, digit_indices)

    digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
    te_pairs, te_y = create_pairs(x_test, digit_indices)
    return input_shape, tr_pairs, tr_y, te_pairs, te_y


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
    return model


def train():
    input_shape, tr_pairs, tr_y, te_pairs, te_y = get_data()
    model = get_model(input_shape)
    if os.path.exists(path='Siamese.h5'):
        model = tf.keras.models.load_model('Siamese.h5')
    assert model.input_shape[0][1:] == input_shape
    rms = RMSprop()
    model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
    cp_checkpoint = ModelCheckpoint(
        filepath='Siamese.h5',
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_freq='epoch'
    )
    with tf.device("/gpu:0"):
        model.fit(
            [tr_pairs[:, 0], tr_pairs[:, 1]],
            tr_y,
            batch_size=128,
            epochs=epochs,
            validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y),
            callbacks=[cp_checkpoint]
        )


if __name__ == '__main__':
    train()
