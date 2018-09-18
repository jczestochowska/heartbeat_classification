import matplotlib.pyplot as plt

from src.signal_utils import get_raw_signal_from_file, repeat_signal_length, downsample_and_filter

LABELS_FILEPATH = '/home/jczestochowska/workspace/heartbeat_classification/data/labels_merged_sets_no_dataset_column.csv'
TEST_DIR_PATH = '/home/jczestochowska/workspace/heartbeat_classification/tests/test_preprocessing_dir'
TEST_FILEPATH = '/home/jczestochowska/workspace/heartbeat_classification/data/set_a/artifact__201012172012.wav'
TEST_FILEPATH1 = '/home/jczestochowska/workspace/heartbeat_classification/data/set_b/extrastole__151_1306779785624_B.wav'
TEST_FILEPATH2 = '/home/jczestochowska/workspace/heartbeat_classification/tests/test_preprocessing_dir/extrastole_128_1306344005749_A.wav'
TEST_LABELS_FILEPATH = '/home/jczestochowska/workspace/heartbeat_classification/tests/test_labels'


data_dir_path = TEST_DIR_PATH
signal_file = TEST_FILEPATH2

signal = get_raw_signal_from_file(signal_file)
signal = repeat_signal_length(signal)
print(len(signal))
f1 = plt.figure(1)
plt.plot(signal)

signal1 = get_raw_signal_from_file(signal_file)
signal1 = repeat_signal_length(signal1)
signal1 = downsample_and_filter(signal1, filter_type='iir', decimate_count=4, sampling_factor=4)
print(len(signal1))
f2 = plt.figure(2)
plt.plot(signal1)

signal2 = get_raw_signal_from_file(signal_file)
signal2 = repeat_signal_length(signal2)
signal2 = downsample_and_filter(signal2, decimate_count=4, sampling_factor=4)
print(len(signal2))
f3 = plt.figure(3)
plt.plot(signal2)


plt.show()
