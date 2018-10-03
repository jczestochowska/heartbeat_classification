import numpy as np
import os
from multiprocessing.pool import Pool

from config import PROJECT_ROOT_DIR
from src.dataset_getters import get_numeric_labels_for_filenames
from src.feature_extraction import get_mfcc

PHYSIONET_DIR = os.path.join(PROJECT_ROOT_DIR, 'data', 'processed', 'preprocessed', 'physionet')
TRAINING_DIR = os.path.join(PHYSIONET_DIR, 'training')
TEST_DIR = os.path.join(PHYSIONET_DIR, 'test')
DESTINATION_DIR = os.path.join(PHYSIONET_DIR, 'serialized')


def serialize_features(filename, data_directory, output_filename):
    features_npy = os.path.join(DESTINATION_DIR, output_filename + '.npy')
    labels_npy = os.path.join(DESTINATION_DIR, output_filename + '_labels' + '.npy')
    labels = np.array(get_numeric_labels_for_filenames(filename))
    filepath = os.path.join(data_directory, filename)
    features = np.array(get_mfcc(filepath))
    flatten_features = np.array(np.ravel(features))
    np.save(features_npy, flatten_features)
    np.save(labels_npy, labels)


if __name__ == '__main__':
    train_filenames = os.listdir(TRAINING_DIR)
    test_filenames = os.listdir(TEST_DIR)
    train_filenames_with_output_names = [[filename] + [TRAINING_DIR, 'train'] for filename in train_filenames]
    test_filenames_with_output_names = [[filename] + [TEST_DIR, 'test'] for filename in test_filenames]
    pool = Pool(5)
    pool.starmap(serialize_features, train_filenames_with_output_names)
    pool.starmap(serialize_features, test_filenames_with_output_names)
