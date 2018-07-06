from unittest import TestCase

import numpy as np

from preprocessing import find_wav_length, find_dataset_longest_wav, repeat_signal_length, get_raw_signal_from_file, \
    downsample_and_filter, create_dataset

TEST_DIR_PATH = '/home/jczestochowska/workspace/heartbeat_classification/tests/test_preprocessing_dir'
TEST_FILEPATH = '/home/jczestochowska/workspace/heartbeat_classification/data/set_a/artifact__201012172012.wav'
TEST_FILEPATH1 = '/home/jczestochowska/workspace/heartbeat_classification/data/set_b/extrastole__151_1306779785624_B.wav'
TEST_FILEPATH2 = '/home/jczestochowska/workspace/heartbeat_classification/tests/test_preprocessing_dir/extrastole_128_1306344005749_A.wav'
TEST_LABELS_FILEPATH = '/home/jczestochowska/workspace/heartbeat_classification/tests/test_labels'


class TestPreprocessing(TestCase):
    def test_find_wav_length(self):
        actual = find_wav_length(TEST_FILEPATH)
        expected = 396900
        self.assertEqual(actual, expected)

    def test_find_wav_length_for_dir(self):
        actual = find_dataset_longest_wav(TEST_DIR_PATH)
        expected = 96640
        self.assertEqual(actual, expected)

    def test_that_repeat_signal_doesnt_change_length(self):
        expected = 3
        actual = len(repeat_signal_length(np.array([1, 2, 3]), expected_length=expected))
        self.assertEqual(actual, expected)

    def test_that_repeat_signal_lengthens(self):
        expected = 8
        actual = len(repeat_signal_length(np.array([1, 2, 3, 4]), expected_length=expected))
        self.assertEqual(actual, expected)

    def test_get_raw_signal_from_file(self):
        actual = len(get_raw_signal_from_file(TEST_FILEPATH1))
        expected = 45233
        self.assertEqual(actual, expected)

    def test_decimating(self):
        signal = [0] * 1024
        signal = np.array(signal)
        actual = len(downsample_and_filter(signal, decimate_count=2, sampling_factor=8))
        expected = 16
        self.assertEqual(expected, actual)

    def test_decimating_exception(self):
        signal = [0] * 22
        downsample_and_filter(signal, 1)
        self.assertRaises(ValueError)
