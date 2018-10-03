from operator import methodcaller
from unittest import TestCase

from src.dataset_getters import get_random_kaggle_filenames_by_label, get_random_physionet_filenames_by_label, \
    get_labels, get_label, map_physionet_label_to_number, map_label_to_string, map_kaggle_label_to_number, \
    get_label_from_filename_processed

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

    def test_get_physionet_labels_column_names(self):
        actual = list(get_labels().columns)
        expected = ['fname', 'label']
        self.assertListEqual(expected, actual)

    def test_get_physionet_label_without_extension(self):
        labels = get_labels()
        audio_filename = 'a0001'
        expected = None
        actual = get_label(audio_filename, labels)
        self.assertEqual(expected, actual)

    def test_get_physionet_label_with_extension(self):
        labels = get_labels()
        audio_filename = 'a0001.wav'
        expected = 1
        actual = get_label(audio_filename, labels)
        self.assertEqual(expected, actual)

    def test_get_physionet_normal_label(self):
        labels = get_labels()
        expected = 1
        audio_filename = get_random_physionet_filenames_by_label(how_many=1, label=expected, set_letter='a')[0]
        actual = get_label(audio_filename, labels)
        self.assertEqual(expected, actual)

    def test_get_physionet_abnormal_label(self):
        labels = get_labels()
        expected = -1
        audio_filename = get_random_physionet_filenames_by_label(how_many=1, label=expected, set_letter='a')[0]
        actual = get_label(audio_filename, labels)
        self.assertEqual(expected, actual)

    def test_get_label_for_non_existing_file(self):
        labels = get_labels()
        actual = get_label(audio_filename="nosuchlabel_01110.wav", labels=labels)
        expected = None
        self.assertEqual(expected, actual)

    def test_get_label(self):
        labels = get_labels()
        actual = get_label(audio_filename="extrahls__201101070953.wav", labels=labels)
        expected = 1
        self.assertEqual(expected, actual)

    def test_map_label_to_number_normal(self):
        actual = map_physionet_label_to_number("normal")
        expected = 0
        self.assertEqual(expected, actual)

    def test_map_label_to_number_abnormal(self):
        actual = map_physionet_label_to_number("abnormal")
        expected = 1
        self.assertEqual(expected, actual)

    def test_map_label_to_number_non_existing_label(self):
        actual = map_physionet_label_to_number("no such label")
        expected = 0
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

    def test_map_label_kaggle_murmur(self):
        label = 'murmur'
        actual = map_kaggle_label_to_number(label)
        expected = -1
        self.assertEqual(expected, actual)

    def test_map_label_kaggle_extrastole(self):
        label = 'extrastole'
        actual = map_kaggle_label_to_number(label)
        expected = -1
        self.assertEqual(expected, actual)

    def test_map_label_kaggle_normal(self):
        label = 'normal'
        actual = map_kaggle_label_to_number(label)
        expected = 1
        self.assertEqual(expected, actual)

    def test_get_label_from_filename_processed_abnormal(self):
        filename = "-1_ihedoichcoiscdcd.wav"
        actual = get_label_from_filename_processed(filename)
        expected = '-1'
        self.assertEqual(expected, actual)

    def test_get_label_from_filename_processed_normal(self):
        filename = "1_ihedoichcoiscdcd.wav"
        actual = get_label_from_filename_processed(filename)
        expected = '1'
        self.assertEqual(expected, actual)

    def test_get_label_from_filename_processed_not_heartbeat(self):
        filename = "0_ihedoichcoiscdcd.wav"
        actual = get_label_from_filename_processed(filename)
        expected = '0'
        self.assertEqual(expected, actual)
