from itertools import product

import numpy as np
import os
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.io import wavfile

from src.dataset_getters import get_kaggle_audio_dir_path, get_kaggle_label, get_physionet_labels, \
    get_random_filenames, map_label_to_number, get_random_physionet_filenames_by_label, get_physionet_label, \
    map_label_to_string, get_random_kaggle_filenames_by_label, get_physionet_audio_dir_path, get_set_name


def plot_violin_plot(title, data, x, y, hue, figsize):
    sns.set(style="whitegrid", palette="pastel", color_codes=True)
    f, ax = plt.subplots(figsize=(figsize, figsize))
    sns.violinplot(x=x, y=y, hue=hue, data=data)
    sns.despine(left=True)
    f.suptitle(title, fontsize=18, fontweight='bold')
    ax.set_xlabel(x, size=16, alpha=0.7)
    ax.set_ylabel(y, size=16, alpha=0.7)
    plt.legend(loc='upper left')


def plot_kaggle_signals_on_square_grid_by_label(set_letter, label, grid_size, figsize):
    indices = list(product(list(range(grid_size)), repeat=2))
    f, ax = plt.subplots(grid_size, grid_size, figsize=(figsize, figsize))
    plt.suptitle(label)
    random_filenames = get_random_kaggle_filenames_by_label(grid_size ** 2, label, set_letter)
    for grid_indices, filename in list(zip(indices, random_filenames)):
        path = os.path.join(get_kaggle_audio_dir_path(set_letter), filename)
        plot_wav_file_on_grid(path, ax, grid_indices)
    plt.show()


def plot_kaggle_signals_by_label(how_many, set_letter, label):
    audio_dir_path = get_kaggle_audio_dir_path(set_letter)
    filenames = get_random_kaggle_filenames_by_label(how_many, label, )
    for filename in filenames:
        path = os.path.join(audio_dir_path, filename)
        plot_kaggle_signal(filename, path)


def plot_kaggle_signals(how_many, set_letter):
    audio_dir_path = get_kaggle_audio_dir_path(set_letter)
    random_filenames = get_random_filenames(how_many, audio_dir_path)
    for filename in random_filenames:
        path = os.path.join(audio_dir_path, filename)
        plot_kaggle_signal(filename, path)


def plot_kaggle_signal(audio_filename, path):
    label = get_kaggle_label(audio_filename)
    plot_wav_file(path, label)


def plot_physionet_signals(how_many, set_letter):
    audio_dir_path = get_physionet_audio_dir_path(set_letter)
    labels = get_physionet_labels()
    random_filenames = get_random_filenames(how_many, audio_dir_path)
    for filename in random_filenames:
        path = os.path.join(audio_dir_path, filename)
        plot_physionet_signal(filename, labels, path)


def plot_physionet_signals_on_square_grid_by_label(set_letter, label, grid_size, figsize):
    indices = list(product(list(range(grid_size)), repeat=2))
    f, ax = plt.subplots(grid_size, grid_size, figsize=(figsize, figsize))
    plt.suptitle(get_set_name(set_letter) + " " + map_label_to_string(label))
    random_physionet_filenames = get_random_physionet_filenames_by_label(grid_size ** 2, label, set_letter)
    for grid_indices, filename in list(zip(indices, random_physionet_filenames)):
        path = os.path.join(get_physionet_audio_dir_path(set_letter), filename)
        plot_wav_file_on_grid(path, ax, grid_indices)
    plt.show()


def plot_physionet_signals_by_label(how_many, set_letter, label):
    label = map_label_to_number(label)
    audio_dir_path = get_physionet_audio_dir_path(set_letter)
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


def plot_wav_file_on_grid(path, grid, grid_coordinates):
    sampling_frequency, signal = wavfile.read(path)
    number_of_samples = len(signal)
    time = np.linspace(0, number_of_samples / sampling_frequency, num=number_of_samples)
    grid[grid_coordinates[0], grid_coordinates[1]].plot(time, signal)
