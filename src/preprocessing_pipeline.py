import numpy as np
import os
import string
import uuid
from multiprocessing.pool import Pool
from scipy.io import wavfile

from config import PROJECT_ROOT_DIR
from src.dataset_getters import get_labels, get_label, map_physionet_label_to_string, get_kaggle_labels_path
from src.subsampling_normalization import get_chunks, downsample_chunks, chunks_magnitude_normalization

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
    multiprocess_files(number_of_sets, labels, destination_dir, dataset)


def multiprocess_files(number_of_sets, labels, destination_dir, dataset):
    set_letters = string.ascii_lowercase[0:number_of_sets]
    arguments = []
    for set_letter in set_letters:
        set_name = 'set_' + set_letter
        source_dir = os.path.join(PROJECT_ROOT_DIR, 'data', 'raw', dataset, set_name)
        filenames = os.listdir(source_dir)
        arguments.extend([[filename] + [source_dir, dataset, labels, destination_dir] for filename in filenames])
    os.mkdir(destination_dir)
    pool = Pool(5)
    pool.starmap(preprocess_file, arguments)


def preprocess_file(filename, source_dir, dataset, labels, destination_dir,
                    new_sampling_rate=4000, chunk_length=5):
    filepath = os.path.join(source_dir, filename)
    sampling_rate, signal = wavfile.read(filepath)
    audio_length = len(signal) // sampling_rate
    signal = signal.tolist()
    if audio_length >= chunk_length:
        chunks = get_chunks(audio_length=audio_length, chunk_length=chunk_length,
                            signal=signal, sampling_rate=sampling_rate)
        if dataset == 'kaggle':
            chunks = downsample_chunks(chunks=chunks, new_sampling_rate=new_sampling_rate)
        normalized_chunks = chunks_magnitude_normalization(chunks=chunks)
        label = get_label(labels=labels, audio_filename=filename)
        if dataset == 'physionet':
            label = map_physionet_label_to_string(label)
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
    dataset = "physionet"
    preprocess_dataset(dataset)
