import numpy as np
import os
from multiprocessing.pool import Pool

from config import PROJECT_ROOT_DIR
from src.dataset_getters import get_label_from_filename_processed, \
    map_physionet_label_to_number
from src.feature_extraction import get_mfcc

PHYSIONET_DIR = os.path.join(PROJECT_ROOT_DIR, 'data', 'processed', 'preprocessed', 'physionet')
TRAINING_DIR = os.path.join(PHYSIONET_DIR, 'training')
TEST_DIR = os.path.join(PHYSIONET_DIR, 'test')
DESTINATION_DIR = os.path.join(PHYSIONET_DIR, 'serialized')


def serialize_features(filename, data_directory):
    label = np.array(map_physionet_label_to_number(get_label_from_filename_processed(filename)))
    filepath = os.path.join(data_directory, filename)
    features = np.array(get_mfcc(filepath))
    flatten_features = np.array(np.ravel(features))
    return flatten_features, label


if __name__ == '__main__':
    train_npy = os.path.join(DESTINATION_DIR, 'train.npy')
    test_npy = os.path.join(DESTINATION_DIR, 'test.npy')
    train_labels_npy = os.path.join(DESTINATION_DIR, 'train_labels' + '.npy')
    test_labels_npy = os.path.join(DESTINATION_DIR, 'test_labels' + '.npy')
    train_filenames = os.listdir(TRAINING_DIR)
    test_filenames = os.listdir(TEST_DIR)
    train_filenames_with_output_names = [[filename] + [TRAINING_DIR] for filename in train_filenames]
    test_filenames_with_output_names = [[filename] + [TEST_DIR] for filename in test_filenames]
    pool = Pool(4)
    train = pool.starmap(serialize_features, train_filenames_with_output_names)
    test = pool.starmap(serialize_features, test_filenames_with_output_names)
    train_features, train_labels = zip(*train)
    test_features, test_labels = zip(*test)
    np.save(train_npy, train_features)
    np.save(train_labels_npy, train_labels)
    np.save(test_npy, test_features)
    np.save(test_labels_npy, test_labels)
