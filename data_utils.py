import csv
import os
import pandas as pd

from signal_utils import prepare_signal_from_file

LABELS_FILEPATH = os.path.join(os.path.split(os.path.abspath(__file__))[0],
                               'data',
                               'labels_merged_sets_no_dataset_column.csv')


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


def create_dataset(data_dir_path, labels_filepath=LABELS_FILEPATH):
    labels_df = pd.read_csv(labels_filepath)
    columns = ['signal', 'label']
    with open('dataset.csv', 'w') as dataset_file:
        csv_writer = csv.writer(dataset_file, delimiter=',')
        csv_writer.writerow(columns)
        for signal_filename in os.listdir(data_dir_path):
            signal_filepath = os.path.join(data_dir_path, signal_filename)
            signal = prepare_signal_from_file(signal_filepath)
            label = get_label(signal_filename, labels_df)
            signal.append(label)
            csv_writer.writerow(signal)


def get_label(signal_filename, labels_df):
    label_series = labels_df.loc[labels_df['fname'] == signal_filename]['label']
    label_values = label_series.values
    return label_series.values[0]
