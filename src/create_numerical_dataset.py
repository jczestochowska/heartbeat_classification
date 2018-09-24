import csv
import logging
import os
import pandas as pd
import string

from config import PROJECT_ROOT_DIR
from src.dataset_getters import get_labels, get_physionet_audio_dir_path, get_label, \
    get_kaggle_audio_dir_path, get_kaggle_labels_path, map_kaggle_label_to_number
from src.signal_utils import prepare_signal_from_file

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def add_kaggle_dataset_to_csv(csv_writer, kaggle_labels):
    set_letters = string.ascii_lowercase[0:2]
    for set_letter in set_letters:
        data_dir_path = get_kaggle_audio_dir_path(set_letter)
        try:
            for signal_filename in os.listdir(data_dir_path):
                label = get_label(signal_filename, kaggle_labels)
                label = map_kaggle_label_to_number(label)
                row = create_row(data_dir_path, signal_filename, label)
                csv_writer.writerow(row)
                LOGGER.warning("File was added: {0}".format(signal_filename))
        except IndexError:
            LOGGER.warning("No label found for file: {0}".format(signal_filename))


def add_physionet_dataset_to_csv(csv_writer, physionet_labels):
    set_letters = string.ascii_lowercase[0:6]
    for set_letter in set_letters:
        data_dir_path = get_physionet_audio_dir_path(set_letter)
        try:
            for signal_filename in os.listdir(data_dir_path):
                label = get_label(signal_filename, physionet_labels)
                row = create_row(data_dir_path, signal_filename, label)
                csv_writer.writerow(row)
                LOGGER.warning("File was added: {0}".format(signal_filename))
        except IndexError:
            LOGGER.warning("No label found for file: {0}".format(signal_filename))


def create_row(data_dir_path, signal_filename, label):
    row = [signal_filename, label]
    signal_filepath = os.path.join(data_dir_path, signal_filename)
    sampling_rate, signal = prepare_signal_from_file(signal_filepath)
    row.append(sampling_rate)
    row.extend(signal)
    return row


if __name__ == '__main__':
    dataset_filename = "numerical_dataset.csv"
    physionet_labels = get_labels()
    kaggle_a_labels = pd.read_csv(get_kaggle_labels_path('a'))
    columns = ['filename', 'label', 'sampling_rate', 'signal']
    dataset_path = os.path.join(PROJECT_ROOT_DIR, "data", "processed", dataset_filename)
    with open(dataset_path, 'a') as dataset_file:
        csv_writer = csv.writer(dataset_file, delimiter=',')
        csv_writer.writerow(columns)
        add_physionet_dataset_to_csv(csv_writer, physionet_labels)
        add_kaggle_dataset_to_csv(csv_writer, kaggle_a_labels)
