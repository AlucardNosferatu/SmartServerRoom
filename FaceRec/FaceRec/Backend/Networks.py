import numpy as np
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Lambda, Dropout, BatchNormalization
from tensorflow.keras.optimizers import RMSprop

from FaceRec.Backend.Configs import num_classes


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


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def awt(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def create_base_net(input_shape, extended_num_classes=None):
    input_layer = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)

    without_dense = Model(input_layer, x)
    if extended_num_classes is not None:
        assert type(extended_num_classes) is int
        output = Dense(extended_num_classes, activation='softmax')(x)
    else:
        output = Dense(num_classes, activation='softmax')(x)
    model = Model(input_layer, output, name='BaseNetwork')
    rms = RMSprop(lr=0.0001)
    model.compile(
        optimizer=rms,
        loss='categorical_crossentropy',
        metrics=['accuracy']
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
    model.compile(loss=contrastive_loss, optimizer=rms, metrics=[awt])

    return model, base_network, without_dense
