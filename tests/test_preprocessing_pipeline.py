import os
import pandas as pd
import shutil
from unittest import TestCase
from unittest.mock import patch

from src.dataset_getters import get_labels
from src.preprocessing_pipeline import preprocess_file, multiprocess_files
from tests.test_signal_utils import TEST_DIR_PATH

TEST_DESTINATION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
TEST_LABELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_labels.csv")


class TestPreprocessingPipeline(TestCase):
    def setUp(self):
        if os.path.exists(TEST_DESTINATION_DIR):
            shutil.rmtree(TEST_DESTINATION_DIR)

    def test_preprocess_file(self):
        os.mkdir(TEST_DESTINATION_DIR)
        test_labels = get_labels(TEST_LABELS_PATH)
        for filename in os.listdir(TEST_DIR_PATH):
            preprocess_file(filename=filename, source_dir=TEST_DIR_PATH, dataset="physionet", labels=test_labels,
                            destination_dir=TEST_DESTINATION_DIR)
        self.assertEqual(len(os.listdir(TEST_DESTINATION_DIR)), 19)

    @patch('src.preprocessing_pipeline.os.path.join')
    @patch('src.preprocessing_pipeline.Pool.starmap')
    def test_multiprocess_files(self, mock_starmap, mock_path):
        mock_path.return_value = TEST_DIR_PATH
        labels = pd.read_csv(TEST_LABELS_PATH)
        multiprocess_files(number_of_sets=2, labels=labels, destination_dir=TEST_DESTINATION_DIR, dataset='test')
        mock_starmap.assert_called_once()
