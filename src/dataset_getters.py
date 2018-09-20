import glob
import logging
import os
import pandas as pd
import random
import re

from config import PROJECT_ROOT_DIR

KAGGLE_PATH = os.path.join(PROJECT_ROOT_DIR, 'data', 'raw', 'kaggle')
PHYSIONET_PATH = os.path.join(PROJECT_ROOT_DIR, 'data', 'raw', 'physionet')

LABELS_FILEPATH = os.path.join(os.path.split(os.path.abspath(__file__))[0],
                               'data',
                               'labels_merged_sets.csv')

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def get_kaggle_labels_path(set_letter):
    return os.path.join(KAGGLE_PATH, 'set_' + set_letter + '.csv')


def get_physionet_labels_path():
    return os.path.join(PHYSIONET_PATH, 'physionet_labels.csv')


def get_audio_dir_path(path, set_letter):
    return os.path.join(path, 'set_' + set_letter)


def get_random_filenames(how_many, directory):
    filenames = os.listdir(directory)
    return random.choices(population=filenames, k=how_many)


def get_random_physionet_filenames_by_label(how_many, label, set_letter):
    labels = get_physionet_labels()
    filenames = labels.loc[labels['label'] == label][labels['filename'].str.match(set_letter)].sample(how_many)[
        'filename'].values
    return list(map(lambda x: x + '.wav', filenames))


def get_physionet_labels():
    labels_path = get_physionet_labels_path()
    return pd.read_csv(labels_path, header=None, names=['filename', 'label'])


def get_physionet_label(audio_filename, labels):
    audio_filename = os.path.splitext(audio_filename)[0]
    return labels.loc[labels['filename'] == audio_filename]['label'].values[0]


def get_random_kaggle_filenames_by_label(how_many, directory, label):
    file_to_find_regex = os.path.join(directory, label)
    file_to_find_regex += '*'
    paths_to_files_filtered_by_labels = glob.glob(file_to_find_regex)
    filenames_filtered_by_labels = list(map(os.path.basename, paths_to_files_filtered_by_labels))
    return random.choices(population=filenames_filtered_by_labels, k=how_many)


def get_set_name(letter):
    return "set_" + letter


def get_kaggle_label(audio_filename):
    return re.search('^[^_]+', audio_filename).group(0)


def map_label_to_string(label):
    return "normal" if label == 1 else "abnormal"


def map_label_to_number(label):
    return 1 if label == 'normal' else -1
