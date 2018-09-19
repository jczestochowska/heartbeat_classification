import os

from config import PROJECT_ROOT_DIR
from src.data_utils import create_dataset

if __name__ == '__main__':
    data_dir_path = os.path.join(PROJECT_ROOT_DIR, 'data/raw/merged_sets')
    labels_filepath = os.path.join(PROJECT_ROOT_DIR, 'data/raw/labels_merged_sets.csv')
    create_dataset(dataset_filename="dataset.csv", data_dir_path=data_dir_path,
                   labels_filepath=labels_filepath)
