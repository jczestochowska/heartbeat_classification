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

DESTINATION_DIR = os.path.join(PROJECT_ROOT_DIR, 'data', 'processed', 'preprocessed')
KAGGLE_DESTINATION_DIR = os.path.join(PROJECT_ROOT_DIR, 'data', 'processed', 'preprocessed', 'kaggle')
PHYSIONET_DESTINATION_DIR = os.path.join(PROJECT_ROOT_DIR, 'data', 'processed', 'preprocessed', 'physionet')


def preprocess_dataset(dataset):
    if dataset == "kaggle":
        number_of_sets = 2
        labels = get_labels(get_kaggle_labels_path())
        destination_dir = KAGGLE_DESTINATION_DIR
    elif dataset == "physionet":
        number_of_sets = 6
        labels = get_labels()
        destination_dir = PHYSIONET_DESTINATION_DIR
    multiprocess_files(number_of_sets, labels, destination_dir)


def multiprocess_files(number_of_sets, labels, destination_dir):
    set_letters = string.ascii_lowercase[0:number_of_sets]
    filenames_with_paths = []
    for set_letter in set_letters:
        set_name = 'set_' + set_letter
        source_dir = os.path.join(PROJECT_ROOT_DIR, 'data', 'raw', 'kaggle', set_name)
        filenames = os.listdir(source_dir)
        filenames_with_paths.extend(list(product(filenames, [source_dir])))
    os.mkdir(destination_dir)
    pool = Pool(5)
    pool.starmap(preprocess_file, filenames_with_paths, dataset, labels, destination_dir)


def preprocess_file(source_dir, filename, dataset, labels, destination_dir=PHYSIONET_DESTINATION_DIR,
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
            new_filename = "{}_{}_{}{}".format(label, dataset, uuid.uuid4().hex, ".wav")
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
    os.mkdir(DESTINATION_DIR)
    dataset = "kaggle"
    preprocess_dataset(dataset)
