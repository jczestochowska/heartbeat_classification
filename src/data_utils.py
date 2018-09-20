import csv
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import re
import wave

from config import PROJECT_ROOT_DIR
from src.signal_utils import prepare_signal_from_file

KAGGLE_PATH = os.path.join(PROJECT_ROOT_DIR, 'data', 'raw', 'kaggle')
PHYSIONET_PATH = os.path.join(PROJECT_ROOT_DIR, 'data', 'raw', 'physionet')

LABELS_FILEPATH = os.path.join(os.path.split(os.path.abspath(__file__))[0],
                               'data',
                               'labels_merged_sets.csv')
LABELS_MAPPING = {'murmur': 1, 'extrastole': 2, 'normal': 3, 'artifact': 4, 'extrahls': 2}

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


def get_random_filenames_by_label(how_many, directory, label):
    file_to_find_regex = os.path.join(directory, label)
    file_to_find_regex += '*'
    filenames_filtered_by_label = glob.glob(file_to_find_regex)
    return random.choices(population=filenames_filtered_by_label, k=how_many)


def get_set_name(letter):
    return "set_" + letter


def get_physionet_label(audio_filename, labels_path):
    labels = pd.read_csv(labels_path)
    return labels.loc[labels['fname'] == audio_filename]['label'].values[0]


def get_kaggle_label(audio_filename):
    return re.search('^[^_]+', audio_filename).group(0)


def map_label(label):
    return "normal" if label == 1 else "abnormal"


def plot_kaggle_signal(audio_filename, set_letter, path=KAGGLE_PATH):
    path = os.path.join(get_audio_dir_path(path, set_letter), audio_filename)
    label = get_kaggle_label(audio_filename)
    audio = wave.open(path, 'rb')
    plot_wav_file(audio, label)


def plot_physionet_signal(audio_filename, set_letter, path=PHYSIONET_PATH):
    path = os.path.join(get_audio_dir_path(path, set_letter), audio_filename)
    label = get_physionet_label(audio_filename, get_kaggle_labels_path(set_letter))
    label = map_label(label)
    audio = wave.open(path, 'rb')
    plot_wav_file(audio, label)


def plot_wav_file(audio, label):
    # Extract Raw Audio from Wav File
    signal = audio.readframes(-1)
    signal = np.frombuffer(signal, 'int16')
    sampling_frequency = audio.getframerate()
    number_of_samples = len(signal)
    time = np.linspace(0, number_of_samples / sampling_frequency, num=number_of_samples)
    length_in_seconds = round(number_of_samples / sampling_frequency)
    description = '\n'.join((
        'sampling frequency {}'.format(sampling_frequency),
        'number of samples {}'.format(number_of_samples),
        'recording length {} s'.format(length_in_seconds)))
    plt.figtext(0.93, 0.5, description, fontsize=13)
    plt.xlabel('time [s]')
    plt.ylabel('amplitude')
    plt.figure(1)
    plt.title(label)
    plt.plot(time, signal)
    plt.show()


def prepare_labels_csv(new_filename="labels_merged_sets_no_dataset_column.csv", labels_filename=LABELS_FILEPATH):
    labels_df = pd.read_csv(labels_filename)
    for index, row in labels_df.iterrows():
        row['fname'] = row['fname'][6:]
        if row['fname'].startswith('Btraining'):
            splitted = row['fname'].split('_')
            if len(splitted) > 5:
                splitted = [splitted[1], splitted[3], splitted[4], splitted[5], splitted[6]]
                row['fname'] = '_'.join(splitted)
            else:
                row['fname'] = row['fname'][10:]
    labels_df.drop('dataset', axis=1, inplace=True)
    saving_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'data', new_filename)
    labels_df.to_csv(saving_path, index=False)


def create_dataset(data_dir_path, dataset_filename="dataset.csv", labels_filepath=LABELS_FILEPATH):
    labels_df = pd.read_csv(labels_filepath)
    columns = ['signal', 'label']
    dataset_path = os.path.join(PROJECT_ROOT_DIR, "data/processed", dataset_filename)
    with open(dataset_path, 'w') as dataset_file:
        csv_writer = csv.writer(dataset_file, delimiter=',')
        csv_writer.writerow(columns)
        try:
            for signal_filename in os.listdir(data_dir_path):
                signal_filepath = os.path.join(data_dir_path, signal_filename)
                signal = prepare_signal_from_file(signal_filepath)
                label = get_label(signal_filename, labels_df)
                signal.append(label)
                csv_writer.writerow(signal)
                LOGGER.warning("File was added: {0}".format(signal_filename))
        except IndexError:
            LOGGER.warning("No label found for file: {0}".format(signal_filename))


def get_label(signal_filename, labels_df):
    return LABELS_MAPPING.get(labels_df.loc[labels_df['fname'] == signal_filename]['label'].values[0])
