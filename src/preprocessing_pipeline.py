import os
import uuid
from multiprocessing.pool import Pool

import numpy as np
from scipy.io import wavfile

from config import PROJECT_ROOT_DIR
from src.data_preparation import get_one_second_chunks, downsample_chunks, chunks_magnitude_normalization
from src.dataset_getters import get_labels, get_label

SOURCE_DIR = os.path.join(PROJECT_ROOT_DIR, 'data', 'raw', 'merged_datasets')
DESTINATION_DIR = os.path.join(PROJECT_ROOT_DIR, 'data', 'processed', 'preprocessed')
LABELS = get_labels()


def preprocess_file(filename, destination_dir=DESTINATION_DIR, labels=LABELS, new_sampling_rate=2000,
                    source_dir=SOURCE_DIR):
    filepath = os.path.join(source_dir, filename)
    sampling_rate, signal = wavfile.read(filepath)
    chunks = get_one_second_chunks(sampling_rate=sampling_rate, signal=signal)
    dowsampled_chunks = downsample_chunks(chunks=chunks, new_sampling_rate=new_sampling_rate)
    normalized_chunks = chunks_magnitude_normalization(chunks=dowsampled_chunks)
    for chunk in normalized_chunks:
        label = get_label(labels=labels, audio_filename=filename)
        if label is not None:
            new_filename = "{}_{}{}".format(label, uuid.uuid4().hex, ".wav")
            new_file_path = os.path.join(destination_dir, new_filename)
            wavfile.write(filename=new_file_path, rate=new_sampling_rate, data=np.array(chunk))


if __name__ == '__main__':
    os.mkdir(DESTINATION_DIR)
    filenames = os.listdir(SOURCE_DIR)
    pool = Pool(5)
    pool.map(preprocess_file, filenames)
