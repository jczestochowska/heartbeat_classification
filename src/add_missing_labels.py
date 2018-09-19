import csv
import os

from src.data_utils import get_kaggle_labels_path, get_audio_dir_path, KAGGLE_PATH, get_kaggle_label

if __name__ == '__main__':
    set_letter = 'a'
    labels_path = get_kaggle_labels_path(set_letter=set_letter)
    set_directory_path = get_audio_dir_path(KAGGLE_PATH, set_letter)
    filenames = os.listdir(set_directory_path)
    with open(labels_path, 'w') as data_csv:
        csvwriter = csv.writer(data_csv, delimiter=',')
        for filename in filenames:
            label = get_kaggle_label(filename)
            csvwriter.writerow([set_letter, filename, label])
