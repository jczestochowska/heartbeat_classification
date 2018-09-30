from itertools import product

import numpy as np
import os
import string
import uuid
from multiprocessing.pool import Pool
from scipy.io import wavfile

from config import PROJECT_ROOT_DIR
from src.data_preparation import get_one_second_chunks, downsample_chunks, chunks_magnitude_normalization
from src.dataset_getters import get_labels, get_label, map_label_to_string, get_kaggle_labels_path

SOURCE_DIR = os.path.join(PROJECT_ROOT_DIR, 'data', 'raw', 'merged_datasets')
PHYSIONET_SOURCE_DIR = os.path.join(PROJECT_ROOT_DIR, 'data', 'raw', 'physionet', 'set_a')
KAGGLE_SOURCE_DIR = os.path.join(PROJECT_ROOT_DIR, 'data', 'raw', 'kaggle', 'set_a')
KAGGLE_DESTINATION_DIR = os.path.join(PROJECT_ROOT_DIR, 'data', 'processed', 'preprocessed', 'kaggle')
DESTINATION_DIR = os.path.join(PROJECT_ROOT_DIR, 'data', 'processed', 'preprocessed')
PHYSIONET_DESTINATION_DIR = os.path.join(PROJECT_ROOT_DIR, 'data', 'processed', 'preprocessed', 'physionet')
LABELS = get_labels()


def preprocess_physionet_file(filename, source_dir, destination_dir=PHYSIONET_DESTINATION_DIR, labels=LABELS,
                              new_sampling_rate=2000):
    filepath = os.path.join(source_dir, filename)
    sampling_rate, signal = wavfile.read(filepath)
    chunks = get_one_second_chunks(sampling_rate=sampling_rate, signal=signal)
    dowsampled_chunks = downsample_chunks(chunks=chunks, new_sampling_rate=new_sampling_rate)
    normalized_chunks = chunks_magnitude_normalization(chunks=dowsampled_chunks)
    label = get_label(labels=labels, audio_filename=filename)
    label = map_label_to_string(label)
    for chunk in normalized_chunks:
        if label is not None:
            new_filename = "{}_{}_{}{}".format(label, "physionet", uuid.uuid4().hex, ".wav")
            new_file_path = os.path.join(destination_dir, new_filename)
            wavfile.write(filename=new_file_path, rate=new_sampling_rate, data=np.array(chunk))


def preprocess_kaggle_file(filename, source_dir, destination_dir=KAGGLE_DESTINATION_DIR, new_sampling_rate=2000):
    if os.path.split(source_dir)[1] == 'set_a':
        labels = get_labels(get_kaggle_labels_path('a'))
    elif os.path.split(source_dir)[1] == 'set_b':
        labels = get_labels(get_kaggle_labels_path('b'))
    filepath = os.path.join(source_dir, filename)
    sampling_rate, signal = wavfile.read(filepath)
    chunks = get_one_second_chunks(sampling_rate=sampling_rate, signal=signal)
    dowsampled_chunks = downsample_chunks(chunks=chunks, new_sampling_rate=new_sampling_rate)
    normalized_chunks = chunks_magnitude_normalization(chunks=dowsampled_chunks)
    label = get_label(labels=labels, audio_filename=filename)
    for chunk in normalized_chunks:
        if label is not None:
            new_filename = "{}_{}_{}{}".format(label, "kaggle", uuid.uuid4().hex, ".wav")
            new_file_path = os.path.join(destination_dir, new_filename)
            wavfile.write(filename=new_file_path, rate=new_sampling_rate, data=np.array(chunk))


def resolve_destination_path(destination_dir, new_filename, label):
    label_dir = "normal"
    if label == 0:
        label_dir = "artifacts"
    elif label == -1:
        label_dir = "abnormal"
    return os.path.join(destination_dir, label_dir, new_filename)


if __name__ == '__main__':
    set_letters = string.ascii_lowercase[0:2]
    filenames_with_paths = []
    for set_letter in set_letters:
        set_name = 'set_' + set_letter
        source_dir = os.path.join(PROJECT_ROOT_DIR, 'data', 'raw', 'kaggle', set_name)
        filenames = os.listdir(source_dir)
        filenames_with_paths.extend(list(product(filenames, [source_dir])))
    os.mkdir(KAGGLE_DESTINATION_DIR)
    pool = Pool(5)
    pool.starmap(preprocess_kaggle_file, filenames_with_paths)
