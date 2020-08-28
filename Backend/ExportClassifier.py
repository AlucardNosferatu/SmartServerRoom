import os

from Backend.Networks import get_model


def export_classifier():
    input_shape = (224, 224, 3)
    extended_num_classes = 4
    model, base_network, without_dense = get_model(
        input_shape,
        extended_num_classes
    )
    if os.path.exists(path='../Models/Siamese.h5'):
        model.load_weights(filepath='../Models/Siamese.h5')
    if os.path.exists(path='../Models/Softmax.h5'):
        base_network.load_weights(filepath='../Models/Softmax.h5')
    if os.path.exists(path='../Models/Conv.h5'):
        without_dense.load_weights(filepath='../Models/Conv.h5')
    base_network.save(filepath='../Models/Classifier.h5')


if __name__ == '__main__':
    export_classifier()
