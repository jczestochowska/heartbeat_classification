import csv
import logging
import os
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import decimate, dlti, butter

LONGEST_AUDIO_LENGTH = 396900
LABELS_FILEPATH = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'data', 'labels_merged_sets_processed.csv')


def find_dataset_longest_wav(data_dir_path):
    the_longest = 0
    for wav_filename in os.listdir(data_dir_path):
        wav_file_path = os.path.join(data_dir_path, wav_filename)
        new_length = find_wav_length(wav_file_path)
        if new_length > the_longest:
            the_longest = new_length
    return the_longest


def find_wav_length(wav_filepath):
    return len(get_raw_signal_from_file(wav_filepath))


def get_raw_signal_from_file(wav_filepath):
    return wavfile.read(wav_filepath)[1]


def repeat_signal_length(initial_signal, expected_length=LONGEST_AUDIO_LENGTH):
    signal = deepcopy(initial_signal)
    if len(signal) != expected_length:
        signal_length = len(initial_signal)
        repetition_number = (expected_length // signal_length) - 1
        repetition_modulo = expected_length % signal_length
        while repetition_number:
            signal = np.hstack((signal, initial_signal))
            repetition_number -= 1
        if repetition_modulo:
            signal = np.hstack((signal, signal[:repetition_modulo]))
    return signal


def downsample_and_filter(signal, decimate_count=1, sampling_factor=4):
    try:
        for _ in range(decimate_count):
            signal = decimate(signal, sampling_factor,
                              ftype=dlti(*butter(8, 0.8 / sampling_factor)), axis=0, zero_phase=True)
    except ValueError:
        logging.warning("signal is too short, cannot downsample with this sampling factor")
    return signal


def prepare_labels_csv(new_filename='labels_merged_sets_processed.csv', labels_filename=LABELS_FILEPATH):
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


def get_label(signal_filename, labels_df):
    label_series = labels_df.loc[labels_df['fname'] == signal_filename]['label']
    return label_series.values[0]


def create_dataset(data_dir_path, labels_filepath=LABELS_FILEPATH):
    labels_df = pd.read_csv(labels_filepath)
    columns = ['signal', 'label']
    with open('dataset.csv', 'w') as dataset_file:
        csv_writer = csv.writer(dataset_file, delimiter=',')
        csv_writer.writerow(columns)
        for signal_filename in os.listdir(data_dir_path):
            signal = get_raw_signal_from_file(os.path.join(data_dir_path, signal_filename))
            signal = repeat_signal_length(signal)
            signal = downsample_and_filter(signal)
            signal = list(map(int, signal))
            label = get_label(signal_filename, labels_df)
            signal.append(label)
            csv_writer.writerow(signal)


if __name__ == '__main__':
    prepare_labels_csv()
