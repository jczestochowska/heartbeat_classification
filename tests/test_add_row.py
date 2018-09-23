from unittest import TestCase

from src.create_numerical_dataset import create_row
from tests.test_preprocessing import TEST_DIR_PATH


class TestCreateNumericalDataset(TestCase):
    def test_create_row(self):
        actual = create_row(TEST_DIR_PATH, 'extrastole__134_1306428161797_C1.wav', 1)
        expected = ['extrastole__134_1306428161797_C1.wav', 1, 4000]
        self.assertEqual(actual[0:3], expected)
