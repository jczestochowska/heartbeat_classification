import glob
import logging
import os
import pandas as pd
import random

from config import PROJECT_ROOT_DIR

KAGGLE_PATH = os.path.join(PROJECT_ROOT_DIR, 'data', 'raw', 'kaggle')
PHYSIONET_PATH = os.path.join(PROJECT_ROOT_DIR, 'data', 'raw', 'physionet')
MERGED_DATASETS_PATH = os.path.join(PROJECT_ROOT_DIR, 'data', 'raw', 'merged_datasets')

LABELS_FILEPATH = os.path.join(PROJECT_ROOT_DIR, 'data', 'raw', 'labels_merged_datasets.csv')
LABELS_MAPPING = {'murmur': -1, 'artifact': 0, 'normal': 1, 'extrastole': -1}
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def get_label_from_filename_processed(filename):
    label = filename[0]
    if label == '-':
        label = '-1'
    return label


def get_kaggle_labels_path(set_letter):
    return os.path.join(KAGGLE_PATH, 'set_' + set_letter + '.csv')


def get_kaggle_labels_path():
    return os.path.join(KAGGLE_PATH, 'labels_kaggle.csv')


def get_physionet_labels_path():
    return os.path.join(PHYSIONET_PATH, 'physionet_labels.csv')


def get_kaggle_audio_dir_path(set_letter):
    return os.path.join(KAGGLE_PATH, 'set_' + set_letter)


def get_physionet_audio_dir_path(set_letter):
    return os.path.join(PHYSIONET_PATH, 'set_' + set_letter)


def get_random_filenames(how_many, directory):
    filenames = os.listdir(directory)
    return random.choices(population=filenames, k=how_many)


def get_random_file_paths(how_many, directory):
    filenames = os.listdir(directory)
    filenames = random.choices(population=filenames, k=how_many)
    return [os.path.join(directory, filename) for filename in filenames]


def get_random_physionet_filenames_by_label(how_many, label, set_letter):
    labels = get_labels()
    return labels.loc[labels['label'] == label][labels['fname'].str.match(set_letter)].sample(how_many)[
        'fname'].values


def get_labels(labels_filepath=LABELS_FILEPATH):
    return pd.read_csv(labels_filepath)


def get_label(audio_filename, labels):
    try:
        label = labels.loc[labels['fname'] == audio_filename]['label'].values[0]
        return label
    except IndexError:
        LOGGER.warning("no label found for file {}".format(audio_filename))


def get_random_kaggle_filenames_by_label(how_many, label, set_letter):
    directory = get_kaggle_audio_dir_path(set_letter)
    file_to_find_regex = os.path.join(directory, label)
    file_to_find_regex += '*'
    paths_to_files_filtered_by_labels = glob.glob(file_to_find_regex)
    filenames_filtered_by_labels = list(map(os.path.basename, paths_to_files_filtered_by_labels))
    return random.choices(population=filenames_filtered_by_labels, k=how_many)


def get_set_name(letter):
    return "set_" + letter


def map_label_to_string(label):
    return "normal" if label == 1 else "abnormal"


def map_physionet_label_to_number(label):
    return 1 if label == 'normal' else -1


def map_kaggle_label_to_number(label):
    return LABELS_MAPPING[label]
