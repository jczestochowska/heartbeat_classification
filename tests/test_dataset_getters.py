from operator import methodcaller
from unittest import TestCase

from src.dataset_getters import get_random_kaggle_filenames_by_label, get_random_physionet_filenames_by_label, \
    get_physionet_labels, get_physionet_label, map_label_to_number, map_label_to_string, get_kaggle_label

SET_LETTER = 'a'

class TestDatasetGetters(TestCase):
    def test_get_random_kaggle_filenames_by_label(self):
        actual = get_random_kaggle_filenames_by_label(how_many=2, label='artifact', set_letter=SET_LETTER)
        expected = 2
        self.assertEqual(expected, len(actual))

    def test_get_random_kaggle_filenames_by_label_noisy(self):
        actual = get_random_kaggle_filenames_by_label(how_many=3, label='murmur', set_letter=SET_LETTER)
        expected = 3
        self.assertEqual(expected, len(actual))

    def test_get_random_kaggle_filenames_by_label_normal(self):
        actual = get_random_kaggle_filenames_by_label(how_many=3, label='normal', set_letter=SET_LETTER)
        expected = 3
        self.assertEqual(expected, len(actual))

    def test_get_random_kaggle_filenames_by_label_extrastole(self):
        actual = get_random_kaggle_filenames_by_label(how_many=2, label='extrastole', set_letter='b')
        expected = 2
        self.assertEqual(expected, len(actual))

    def test_get_random_physionet_filenames_by_label(self):
        actual = len(get_random_physionet_filenames_by_label(3, 1, 'a'))
        expected = 3
        self.assertEqual(expected, actual)

    def test_test_get_random_physionet_filenames_by_label_should_add_extension(self):
        actual_list = get_random_physionet_filenames_by_label(3, -1, 'b')
        actual = all(element == True for element in list(map(methodcaller("endswith", ".wav"), actual_list)))
        self.assertTrue(actual)

    def test_get_physionet_labels_adds_columnnames(self):
        actual = list(get_physionet_labels().columns)
        expected = ['filename', 'label']
        self.assertListEqual(expected, actual)

    def test_get_physionet_label_without_extension(self):
        labels = get_physionet_labels()
        audio_filename = 'a0001'
        expected = 1
        actual = get_physionet_label(audio_filename, labels)
        self.assertEqual(expected, actual)

    def test_get_physionet_label_with_extension(self):
        labels = get_physionet_labels()
        audio_filename = 'a0001.wav'
        expected = 1
        actual = get_physionet_label(audio_filename, labels)
        self.assertEqual(expected, actual)

    def test_get_physionet_normal_label(self):
        labels = get_physionet_labels()
        expected = 1
        audio_filename = get_random_physionet_filenames_by_label(how_many=1, label=expected, set_letter='a')[0]
        actual = get_physionet_label(audio_filename, labels)
        self.assertEqual(expected, actual)

    def test_get_physionet_abnormal_label(self):
        labels = get_physionet_labels()
        expected = -1
        audio_filename = get_random_physionet_filenames_by_label(how_many=1, label=expected, set_letter='a')[0]
        actual = get_physionet_label(audio_filename, labels)
        self.assertEqual(expected, actual)

    def test_get_kaggle_label_from_filename_murmur(self):
        actual = get_kaggle_label(audio_filename="murmur_01110.wav")
        expected = 'murmur'
        self.assertEqual(expected, actual)

    def test_get_kaggle_label_from_filename_wrong_filename(self):
        actual = get_kaggle_label(audio_filename="nosuchlabel_01110.wav")
        expected = 'nosuchlabel'
        self.assertEqual(expected, actual)

    def test_map_label_to_number_normal(self):
        actual = map_label_to_number("normal")
        expected = 1
        self.assertEqual(expected, actual)

    def test_map_label_to_number_abnormal(self):
        actual = map_label_to_number("abnormal")
        expected = -1
        self.assertEqual(expected, actual)

    def test_map_label_to_number_non_existing_label(self):
        actual = map_label_to_number("no such label")
        expected = -1
        self.assertEqual(expected, actual)

    def test_map_label_to_string_normal(self):
        actual = map_label_to_string(1)
        expected = "normal"
        self.assertEqual(expected, actual)

    def test_map_label_to_string_abnormal(self):
        actual = map_label_to_string(-1)
        expected = "abnormal"
        self.assertEqual(expected, actual)

    def test_map_label_to_string_non_existing_label(self):
        actual = map_label_to_string(10)
        expected = "abnormal"
        self.assertEqual(expected, actual)
