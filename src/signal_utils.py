import logging
import numpy as np
import os
from copy import deepcopy
from scipy.io import wavfile
from scipy.signal import decimate, dlti, butter, cheby1

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


def prepare_signal_from_file(signal_filepath):
    sampling_rate, signal = wavfile.read(signal_filepath)
    signal = repeat_signal_length(signal)
    signal = list(map(int, signal))
    return sampling_rate, signal


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


def resolve_filter(filter_type, sampling_factor):
    if filter_type == "butterworth":
        return dlti(*butter(8, 0.8 / sampling_factor))
    elif filter_type == "chebyshev":
        return dlti(*cheby1(8, 0.05, 0.8 / sampling_factor))
    raise UnknownFilterException()


def decimate_(signal, filter_type="butterworth", decimate_count=1, sampling_factor=4):
    filter_ = resolve_filter(filter_type, sampling_factor)
    try:
        for _ in range(decimate_count):
            signal = decimate(signal, sampling_factor, ftype=filter_, axis=0, zero_phase=True)
    except ValueError:
        logging.warning("signal is too short, cannot downsample with this sampling factor")
    except UnknownFilterException():
        logging.warning("chosen filter cannot be resolved, supported are chebyshev or butterworth")
    return signal


class UnknownFilterException(Exception):
    pass
