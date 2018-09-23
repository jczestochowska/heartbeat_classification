import numpy as np
import os
from unittest import TestCase

from config import PROJECT_ROOT_DIR
from src.signal_utils import find_wav_length, find_dataset_longest_wav, repeat_signal_length, get_raw_signal_from_file, \
    decimate_, prepare_signal_from_file

TEST_DIR_PATH = os.path.join(PROJECT_ROOT_DIR, 'tests', 'test_preprocessing_dir')
TEST_FILEPATH = os.path.join(PROJECT_ROOT_DIR, 'data', 'raw', 'kaggle', 'set_a', 'artifact__201012172012.wav')
TEST_FILEPATH1 = os.path.join(PROJECT_ROOT_DIR,
                              'data', 'raw', 'kaggle', 'set_b', 'extrastole__151_1306779785624_B.wav')
TEST_FILEPATH2 = os.path.join(PROJECT_ROOT_DIR,
                              'tests', 'test_preprocessing_dir', 'extrastole_128_1306344005749_A.wav')
TEST_LABELS_FILEPATH = os.path.join(PROJECT_ROOT_DIR, 'tests', 'test_preprocessing_dir', 'test_labels')


class TestPreprocessing(TestCase):
    def test_find_wav_length(self):
        actual = find_wav_length(TEST_FILEPATH)
        expected = 396900
        self.assertEqual(actual, expected)

    def test_find_wav_length_for_dir(self):
        actual = find_dataset_longest_wav(TEST_DIR_PATH)
        expected = 396900
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
        actual = len(decimate_(signal, decimate_count=2, sampling_factor=8))
        expected = 16
        self.assertEqual(expected, actual)

    def test_decimating_exception(self):
        signal = [0] * 22
        decimate_(signal, sampling_factor=1)
        self.assertRaises(ValueError)

    def test_prepare_signal_from_file(self):
        actual = prepare_signal_from_file(TEST_FILEPATH)[1]
        self.assertTrue(all(isinstance(number, int) for number in actual))
