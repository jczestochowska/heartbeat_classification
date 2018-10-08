import numpy as np
import scipy
from unittest import TestCase

from src.subsampling_normalization import get_chunks, downsample_chunks, chunks_magnitude_normalization
from tests.test_signal_utils import TEST_FILEPATH


class TestSubsampling(TestCase):
    def test_get_chunks_chunks_have_correct_length(self):
        # given
        chunk_length = 5
        # when
        fs, signal = scipy.io.wavfile.read(TEST_FILEPATH)
        signal = signal.tolist()
        actual = get_chunks(chunk_length, signal, fs, len(signal) // fs)
        # then
        actual = all(element == chunk_length * fs for element in list(map(len, actual)))
        self.assertTrue(actual)

    def test_get_chunks_returns_correct_number_of_chunks(self):
        # given
        fs, signal = scipy.io.wavfile.read(TEST_FILEPATH)
        signal = signal.tolist()
        audio_length = len(signal) // fs
        chunk_length = 5
        chunks_number = round(audio_length / chunk_length)
        # when
        actual = get_chunks(chunk_length=chunk_length, signal=signal, sampling_rate=fs, audio_length=audio_length)
        # then
        self.assertEqual(len(actual), chunks_number)

    def test_downsample_chunks(self):
        data = [np.linspace(0, 100, 3000), np.linspace(0, 100, 3000)]
        actual = downsample_chunks(data)
        expected = 2000
        actual = all(element == expected for element in list(map(len, actual)))
        self.assertTrue(actual)

    def test_chunks_magnitude_normalization(self):
        chunks = [
            [1, 0, 0], [3, 2, -1]
        ]
        actual = chunks_magnitude_normalization(chunks)
        expected = [
            [1.4142135623730951, -0.7071067811865475, -0.7071067811865475],
            [0.9805806756909202, 0.3922322702763681, -1.372812945967288]
        ]
        self.assertEqual(len(expected), len(actual))
        self.assertListEqual(expected[0], actual[0])
        self.assertListEqual(expected[1], actual[1])
