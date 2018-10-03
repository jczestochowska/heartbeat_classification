import os
import random
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
    num_normal_samples = len(normal_samples)
    all_abnormal_samples = os.listdir(PHYSIONET_ABNORMAL_DIR)
    abnormal_samples = random.choices(all_abnormal_samples, k=num_normal_samples)
    normal_divided = train_test_split(normal_samples, train_size=0.7)
    abnormal_divided = train_test_split(abnormal_samples, train_size=0.7)
    for filename in normal_divided[0]:
        shutil.copy(os.path.join(PHYSIONET_NORMAL_DIR, filename), TRAINING_DIR)
    for filename in abnormal_divided[0]:
        shutil.copy(os.path.join(PHYSIONET_ABNORMAL_DIR, filename), TRAINING_DIR)
    for filename in normal_divided[1]:
        shutil.copy(os.path.join(PHYSIONET_NORMAL_DIR, filename), TEST_DIR)
    for filename in abnormal_divided[1]:
        shutil.copy(os.path.join(PHYSIONET_ABNORMAL_DIR, filename), TEST_DIR)
