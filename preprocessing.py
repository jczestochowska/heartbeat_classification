import os
import wave
from copy import deepcopy

import numpy as np
from scipy.signal import decimate

LONGEST_AUDIO_LENGTH = 396900


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
    raw_signal = wave.open(wav_filepath, 'rb')
    initial_signal = raw_signal.readframes(-1)
    return np.frombuffer(initial_signal, 'int16')


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


def decimating(signal, decimate_count, sampling_factor=8):
    for _ in range(decimate_count):
        signal = decimate(signal, sampling_factor, axis=0, zero_phase=True)
    return signal
