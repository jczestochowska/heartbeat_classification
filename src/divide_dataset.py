import os
import shutil
from sklearn.model_selection import train_test_split

from config import PROJECT_ROOT_DIR

PHYSIONET_NORMAL_DIR = os.path.join(PROJECT_ROOT_DIR, 'data', 'processed', 'preprocessed', 'physionet', 'normal')
PHYSIONET_ABNORMAL_DIR = os.path.join(PROJECT_ROOT_DIR, 'data', 'processed', 'preprocessed', 'physionet', 'abnormal')
PHYSIONET_DIR = os.path.join(PROJECT_ROOT_DIR, 'data', 'processed', 'preprocessed', 'physionet')
TRAINING_DIR = os.path.join(PHYSIONET_DIR, 'training')
TEST_DIR = os.path.join(PHYSIONET_DIR, 'test')

if __name__ == '__main__':
    os.mkdir(TRAINING_DIR)
    os.mkdir(TEST_DIR)
    normal_samples = os.listdir(PHYSIONET_NORMAL_DIR)
    abnormal_samples = os.listdir(PHYSIONET_ABNORMAL_DIR)
    dataset = normal_samples + abnormal_samples
    divided = train_test_split(dataset, train_size=0.7)
    for filename in divided[0]:
        if filename.split('_')[0] == 'abnormal':
            directory = PHYSIONET_ABNORMAL_DIR
        else:
            directory = PHYSIONET_NORMAL_DIR
        shutil.copy(os.path.join(directory, filename), TRAINING_DIR)
    for filename in divided[1]:
        if filename.split('_')[0] == 'abnormal':
            directory = PHYSIONET_ABNORMAL_DIR
        else:
            directory = PHYSIONET_NORMAL_DIR
        shutil.copy(os.path.join(directory, filename), TEST_DIR)
