import os
import shutil
from unittest import TestCase

from src.dataset_getters import get_labels
from src.preprocessing_pipeline import preprocess_file
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
            preprocess_file(source_dir=TEST_DIR_PATH, filename=filename, labels=test_labels, dataset="physionet",
                            destination_dir=TEST_DESTINATION_DIR)
        self.assertEqual(len(os.listdir(TEST_DESTINATION_DIR)), 102)
