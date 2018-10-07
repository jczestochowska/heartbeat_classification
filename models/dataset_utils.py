import numpy as np
import os
import shutil
import tensorflow as tf

from config import PROJECT_ROOT_DIR
from models.logistic_regression import SUMMARIES_DIR

TRAIN = os.path.join(PROJECT_ROOT_DIR, 'data', 'processed', 'preprocessed', 'physionet', 'serialized', 'train.npy')
TEST = os.path.join(PROJECT_ROOT_DIR, 'data', 'processed', 'preprocessed', 'physionet', 'serialized', 'test.npy')
TRAIN_LABELS = os.path.join(PROJECT_ROOT_DIR, 'data', 'processed', 'preprocessed', 'physionet', 'serialized',
                            'train_labels.npy')
TEST_LABELS = os.path.join(PROJECT_ROOT_DIR, 'data', 'processed', 'preprocessed', 'physionet', 'serialized',
                           'test_labels.npy')


def load_dataset():
    return np.load(TRAIN), np.load(TEST), np.load(TRAIN_LABELS), np.load(TEST_LABELS)


def get_balanced_dataset(train_features, train_labels):
    normal_data = np.squeeze(train_features[np.argwhere(train_labels == 0)], axis=1)
    abnormal_data = np.squeeze(train_features[np.argwhere(train_labels == 1)], axis=1)

    repeat = len(abnormal_data) - len(normal_data)
    normal_idx = np.argwhere(train_labels == 0).tolist()
    normal_data = train_features[np.concatenate((normal_idx, normal_idx[0:repeat]), axis=0)]
    normal_data = np.squeeze(normal_data, axis=1)

    normal_labels = np.zeros(shape=(normal_data.shape[0]))
    abnormal_labels = np.ones(shape=(abnormal_data.shape[0]))

    dataset_abnormal = tf.data.Dataset.from_tensor_slices((abnormal_data, abnormal_labels))
    dataset_normal = tf.data.Dataset.from_tensor_slices((normal_data, normal_labels))
    dataset = dataset_normal.concatenate(dataset_abnormal)
    return dataset.shuffle(buffer_size=train_features.shape[0])


def delete_tensorboard_summaries():
    train_summary = os.path.join(SUMMARIES_DIR, 'train')
    test_summary = os.path.join(SUMMARIES_DIR, 'test')
    if os.path.exists(train_summary) and os.path.exists(test_summary):
        shutil.rmtree(train_summary)
        shutil.rmtree(test_summary)
