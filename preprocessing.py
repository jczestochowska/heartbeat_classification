import os
import wave

import numpy as np


def find_wav_length(filepath):
    raw_signal = wave.open(filepath, 'rb')
    signal = raw_signal.readframes(-1)
    signal = np.frombuffer(signal, 'int16')
    return len(signal)


def find_dataset_longest_wav(data_dir_path):
    the_longest = 0
    for wav_filename in os.listdir(data_dir_path):
        wav_file_path = os.path.join(data_dir_path, wav_filename)
        new_length = find_wav_length(wav_file_path)
        if new_length > the_longest:
            the_longest = new_length
    return the_longest

