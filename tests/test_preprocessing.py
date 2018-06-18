import numpy as np
from unittest import TestCase

from preprocessing import find_wav_length, find_dataset_longest_wav, repeat_signal_length, get_raw_signal_from_file

TEST_DIR_PATH = '/home/jczestochowska/workspace/heartbeat_classification/tests/test_preprocessing_dir'
TEST_FILEPATH = '/home/jczestochowska/workspace/heartbeat_classification/data/set_a/artifact__201012172012.wav'
TEST_FILEPATH1 = '/home/jczestochowska/workspace/heartbeat_classification/data/set_b/extrastole__151_1306779785624_B.wav'


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
