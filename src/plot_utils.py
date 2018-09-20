import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.io import wavfile

from src.dataset_getters import KAGGLE_PATH, get_audio_dir_path, get_kaggle_label, PHYSIONET_PATH, get_physionet_labels, \
    get_random_filenames, map_label_to_number, get_random_physionet_filenames_by_label, get_physionet_label, \
    map_label_to_string


def plot_kaggle_signal(audio_filename, set_letter, path=KAGGLE_PATH):
    path = os.path.join(get_audio_dir_path(path, set_letter), audio_filename)
    label = get_kaggle_label(audio_filename)
    plot_wav_file(path, label)


def plot_physionet_signals(how_many, set_letter, path=PHYSIONET_PATH):
    audio_dir_path = get_audio_dir_path(path, set_letter)
    labels = get_physionet_labels()
    random_filenames = get_random_filenames(how_many, audio_dir_path)
    for filename in random_filenames:
        path = os.path.join(audio_dir_path, filename)
        plot_physionet_signal(filename, labels, path)


def plot_physionet_signals_by_label(how_many, set_letter, label, path=PHYSIONET_PATH):
    label = map_label_to_number(label)
    audio_dir_path = get_audio_dir_path(path, set_letter)
    labels = get_physionet_labels()
    random_filenames = get_random_physionet_filenames_by_label(how_many, label, set_letter)
    for filename in random_filenames:
        path = os.path.join(audio_dir_path, filename)
        plot_physionet_signal(filename, labels, path)


def plot_physionet_signal(audio_filename, labels, path):
    label = get_physionet_label(audio_filename, labels)
    label = map_label_to_string(label)
    plot_wav_file(path, label)


def plot_wav_file(path, label):
    sampling_frequency, signal = wavfile.read(path)
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
