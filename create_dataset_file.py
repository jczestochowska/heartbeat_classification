import os
import pandas as pd

from config import PROJECT_ROOT_DIR
from data_utils import create_dataset

if __name__ == '__main__':
    data_dir_path = os.path.join(PROJECT_ROOT_DIR, 'data/merged_sets')
    labels_filepath = os.path.join(PROJECT_ROOT_DIR, 'data/labels_merged_sets_no_dataset_column.csv')
    create_dataset(data_dir_path=data_dir_path,
                   labels_filepath=labels_filepath)
