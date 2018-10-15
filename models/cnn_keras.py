import keras_metrics
import numpy as np
from keras import callbacks
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv1D, MaxPool1D, BatchNormalization, Dense, GlobalMaxPool1D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import np_utils

from models.dataset_utils import load_dataset


def prepare_dataset():
    train_features, test_features, train_labels, test_labels = load_dataset(mfcc=False)
    train_features = train_features[:, :, np.newaxis]
    test_features = test_features[:, :, np.newaxis]
    train_labels = np_utils.to_categorical(train_labels)
    test_labels = np_utils.to_categorical(test_labels)
    return train_features, test_features, train_labels, test_labels


def get_keras_cnn(convo_input_shape):
    model = Sequential()
    model.add(Conv1D(filters=5, kernel_size=10, activation='relu',
                     input_shape=convo_input_shape,
                     kernel_regularizer=l2(0.025)))
    model.add(MaxPool1D(strides=5))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=5, kernel_size=10, activation='relu',
                     kernel_regularizer=l2(0.05)))
    model.add(MaxPool1D(strides=4))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=10, kernel_size=10, activation='relu',
                     kernel_regularizer=l2(0.1)))
    model.add(MaxPool1D(strides=5))
    model.add(Conv1D(filters=64, kernel_size=10))
    model.add(BatchNormalization())
    model.add(GlobalMaxPool1D())
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dense(2, activation='softmax'))
    return model


def batch_generator(x_train, y_train, batch_size):
    x_batch = np.empty((batch_size, x_train.shape[1], x_train.shape[2]), dtype='float32')
    y_batch = np.empty((batch_size, y_train.shape[1]), dtype='float32')
    full_idx = range(x_train.shape[0])

    while True:
        batch_idx = np.random.choice(full_idx, batch_size)
        x_batch = x_train[batch_idx]
        y_batch = y_train[batch_idx]

        for i in range(batch_size):
            sz = np.random.randint(x_batch.shape[1])
            x_batch[i] = np.roll(x_batch[i], sz, axis=0)

        yield x_batch, y_batch


if __name__ == '__main__':
    batch_size = 20
    learning_rate = 1e-4
    train_features, test_features, train_labels, test_labels = prepare_dataset()
    convo_input_shape = train_features.shape[1:]
    model = get_keras_cnn(convo_input_shape)
    model.compile(optimizer=Adam(learning_rate), loss='categorical_crossentropy',
                  metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall()])
    weight_saver = ModelCheckpoint('convo_weights.h5', monitor='val_loss',
                                   save_best_only=True, save_weights_only=False)

    tensorboard = callbacks.TensorBoard(log_dir='./tensorboard_summaries', histogram_freq=0, batch_size=1,
                                        write_graph=True,
                                        write_grads=False, write_images=False, embeddings_freq=0,
                                        embeddings_layer_names=None,
                                        embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

    hist = model.fit_generator(batch_generator(train_features, train_labels, batch_size),
                               epochs=50, steps_per_epoch=1000,
                               validation_data=(test_features, test_labels),
                               callbacks=[weight_saver, tensorboard],
                               verbose=2)
