import wave
from unittest import TestCase

import numpy as np

from preprocessing import find_wav_length, find_dataset_longest_wav


class TestPreprocessing(TestCase):
    def setUp(self):
        self.test_dir_path = '/home/jczestochowska/workspace/heartbeat_classification/tests/test_preprocessing_dir'

    def test_find_wav_length(self):
        test_filepath = '/home/jczestochowska/workspace/heartbeat_classification/data/set_a/artifact__201012172012.wav'
        actual = find_wav_length(test_filepath)
        expected = 396900
        self.assertEqual(actual, expected)

    def test_find_wav_length_for_dir(self):
        actual = find_dataset_longest_wav(self.test_dir_path)
        expected = 96640
        self.assertEqual(actual, expected)





