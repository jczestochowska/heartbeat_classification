import numpy as np
import os
import shutil
import tensorflow as tf

from config import PROJECT_ROOT_DIR

SUMMARIES_DIR = os.path.join(PROJECT_ROOT_DIR, 'models', 'tensorboard_summaries')
TRAIN = os.path.join(PROJECT_ROOT_DIR, 'data', 'processed', 'preprocessed', 'physionet',
                     'serialized', 'mfcc', 'train.npy')
TEST = os.path.join(PROJECT_ROOT_DIR, 'data', 'processed', 'preprocessed', 'physionet',
                    'serialized', 'mfcc', 'test.npy')
TRAIN_LABELS = os.path.join(PROJECT_ROOT_DIR, 'data', 'processed', 'preprocessed', 'physionet',
                            'serialized', 'mfcc', 'train_labels.npy')
TEST_LABELS = os.path.join(PROJECT_ROOT_DIR, 'data', 'processed', 'preprocessed', 'physionet',
                           'serialized', 'mfcc', 'test_labels.npy')
TRAIN_RAW = os.path.join(PROJECT_ROOT_DIR, 'data', 'processed', 'preprocessed', 'physionet',
                         'serialized', 'no_feature_extraction', 'train.npy')
TEST_RAW = os.path.join(PROJECT_ROOT_DIR, 'data', 'processed', 'preprocessed', 'physionet',
                        'serialized', 'no_feature_extraction', 'test.npy')
TRAIN_LABELS_RAW = os.path.join(PROJECT_ROOT_DIR, 'data', 'processed', 'preprocessed', 'physionet',
                                'serialized', 'no_feature_extraction', 'train_labels.npy')
TEST_LABELS_RAW = os.path.join(PROJECT_ROOT_DIR, 'data', 'processed', 'preprocessed', 'physionet', 'serialized',
                               'no_feature_extraction', 'test_labels.npy')


def load_dataset(mfcc=True):
    if mfcc:
        return np.load(TRAIN), np.load(TEST), np.load(TRAIN_LABELS), np.load(TEST_LABELS)
    else:
        return np.load(TRAIN_RAW), np.load(TEST_RAW), np.load(TRAIN_LABELS_RAW), np.load(TEST_LABELS_RAW)


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


def create_balanced_dataset_batch(batch_size):
    train_features, test_features, train_labels, test_labels = load_dataset(mfcc=False)
    balanced_dataset = get_balanced_dataset(train_features, train_labels)
    balanced_dataset = balanced_dataset.repeat().batch(batch_size)
    return balanced_dataset, test_features, test_labels
