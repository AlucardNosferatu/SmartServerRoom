import random

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Lambda
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from Configs import epochs, batch_size
from Data import load_4_faces
from Networks import create_base_net, euclid_dis, eucl_dist_output_shape


def two_losses(y_true, y_pred):
    margin = 1
    y_p = y_pred[0]
    y_a_p = y_pred[1]
    y_b_p = y_pred[2]
    y_t = y_true[0]
    y_a_t = y_true[1]
    y_b_t = y_true[2]
    a_class_loss = K.categorical_crossentropy(y_a_t, y_a_p)
    b_class_loss = K.categorical_crossentropy(y_b_t, y_b_p)
    square_pred = K.square(y_p)
    margin_square = K.square(K.maximum(margin - y_p, 0))
    contrastive = K.mean(y_t * square_pred + (1 - y_t) * margin_square)
    return contrastive + a_class_loss + b_class_loss


def get_model_2_outputs(input_shape=(224, 224, 3), extended_num_classes=None):
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

    model = Model([input_a, input_b], [distance, processed_a, processed_b])
    rms = RMSprop(lr=0.0001)
    model.output_names[0] = 'AB'
    model.output_names[1] = 'PA'
    model.output_names[2] = 'PB'
    model.compile(loss=two_losses, optimizer=rms, metrics=['acc'])

    return model, base_network, without_dense


def create_pairs_with_labels(x, y_one_hot, digit_indices, extended_num_classes=None):
    for i in range(len(digit_indices)):
        di = digit_indices[i].tolist()
        temp = di.copy()
        while len(di) < 10:
            di.append(random.choice(temp))
        digit_indices[i] = np.array(di)
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(extended_num_classes)]) - 1
    for d in range(extended_num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, extended_num_classes)
            dn = (d + inc) % extended_num_classes
            z3, z4 = digit_indices[d][i], digit_indices[dn][i]
            assert z1 == z3
            pairs += [[x[z3], x[z4]]]
            labels += [[1, y_one_hot[z1], y_one_hot[z2], ], [0, y_one_hot[z3], y_one_hot[z4]]]
    return np.array(pairs), np.array(labels)


def get_data_2_labels():
    tr, te = load_4_faces(extended_num_classes=4)
    xtr, ytr = tr
    xte, yte = te
    y_train_one_hot = to_categorical(ytr, num_classes=4)
    y_test_one_hot = to_categorical(yte, num_classes=4)

    digit_indices = [np.where(ytr == i)[0] for i in range(4)]
    tr_pairs, tr_y = create_pairs_with_labels(xtr, y_train_one_hot, digit_indices, 4)
    digit_indices = [np.where(yte == i)[0] for i in range(4)]
    te_pairs, te_y = create_pairs_with_labels(xte, y_test_one_hot, digit_indices, 4)
    return tr_pairs, tr_y, te_pairs, te_y


def train_with_2_losses():
    m, bn, wd = get_model_2_outputs(extended_num_classes=4)
    trp, tra_y, tep, tes_y = get_data_2_labels()
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
    m.fit(
        x=[trp[:, 0], trp[:, 1]],
        y=[
            np.array(tra_y[:, 0].tolist()),
            np.array(tra_y[:, 1].tolist()),
            np.array(tra_y[:, 2].tolist())
        ],
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(
            [tep[:, 0], tep[:, 1]],
            [
                np.array(tes_y[:, 0].tolist()),
                np.array(tes_y[:, 1].tolist()),
                np.array(tes_y[:, 2].tolist())
            ]
        ),
        validation_steps=2,
        callbacks=[
            cp_checkpoint,
            es_checkpoint
        ]
    )
    m.save_weights(filepath='Models/Siamese.h5')
    bn.save_weights(filepath='Models/Softmax.h5')
    wd.save_weights(filepath='Models/Conv.h5')


if __name__ == '__main__':
    train_with_2_losses()
