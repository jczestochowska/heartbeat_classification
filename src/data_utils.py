import csv
import logging
import os
import pandas as pd

from config import PROJECT_ROOT_DIR
from src.signal_utils import prepare_signal_from_file

LABELS_FILEPATH = os.path.join(os.path.split(os.path.abspath(__file__))[0],
                               'data',
                               'labels_merged_sets_no_dataset_column.csv')
LABELS_MAPPING = {'murmur': 1, 'extrastole': 2, 'normal': 3, 'artifact': 4, 'extrahls': 2}

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def prepare_labels_csv(new_filename="labels_merged_sets_no_dataset_column.csv", labels_filename=LABELS_FILEPATH):
    labels_df = pd.read_csv(labels_filename)
    for index, row in labels_df.iterrows():
        row['fname'] = row['fname'][6:]
        if row['fname'].startswith('Btraining'):
            splitted = row['fname'].split('_')
            if len(splitted) > 5:
                splitted = [splitted[1], splitted[3], splitted[4], splitted[5], splitted[6]]
                row['fname'] = '_'.join(splitted)
            else:
                row['fname'] = row['fname'][10:]
    labels_df.drop('dataset', axis=1, inplace=True)
    saving_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'data', new_filename)
    labels_df.to_csv(saving_path, index=False)


def create_dataset(data_dir_path, dataset_filename="dataset.csv", labels_filepath=LABELS_FILEPATH):
    labels_df = pd.read_csv(labels_filepath)
    columns = ['signal', 'label']
    dataset_path = os.path.join(PROJECT_ROOT_DIR, "data/processed", dataset_filename)
    with open(dataset_path, 'w') as dataset_file:
        csv_writer = csv.writer(dataset_file, delimiter=',')
        csv_writer.writerow(columns)
        try:
            for signal_filename in os.listdir(data_dir_path):
                signal_filepath = os.path.join(data_dir_path, signal_filename)
                signal = prepare_signal_from_file(signal_filepath)
                label = get_label(signal_filename, labels_df)
                signal.append(label)
                csv_writer.writerow(signal)
                LOGGER.warning("File was added: {0}".format(signal_filename))
        except IndexError:
            LOGGER.warning("No label found for file: {0}".format(signal_filename))


def get_label(signal_filename, labels_df):
    return LABELS_MAPPING.get(labels_df.loc[labels_df['fname'] == signal_filename]['label'].values[0])
