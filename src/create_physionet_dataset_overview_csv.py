import csv
import os
import string
from scipy.io import wavfile

from config import PROJECT_ROOT_DIR
from src.dataset_getters import get_physionet_audio_dir_path, get_labels, get_label, get_set_name, \
    map_physionet_label_to_string

if __name__ == '__main__':
    set_letters = list(string.ascii_lowercase[0:6])
    labels = get_labels()
    csv_filename = 'physionet_dataset_overview.csv'
    saving_path = os.path.join(PROJECT_ROOT_DIR, 'data', 'processed', csv_filename)
    with open(saving_path, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(
            ['filename', 'recording_length', 'sampling_frequency', 'magnitude_bandwidth', 'max_magnitude',
             'mean_magnitude', 'label',
             'set'])
        for set_letter in set_letters:
            audio_dir_path = get_physionet_audio_dir_path(set_letter=set_letter)
            filenames = os.listdir(audio_dir_path)
            for filename in filenames:
                file_path = os.path.join(audio_dir_path, filename)
                label = get_label(filename, labels)
                label = map_physionet_label_to_string(label)
                sampling_frequency, signal = wavfile.read(file_path)
                recording_length = round(len(signal) / sampling_frequency)
                magnitude_bandwidth = max(signal) - min(signal)
                mean_magnitude = sum(signal) / len(signal)
                writer.writerow(
                    [filename, recording_length, sampling_frequency, magnitude_bandwidth, max(signal), mean_magnitude,
                     label,
                                 get_set_name(set_letter)])
