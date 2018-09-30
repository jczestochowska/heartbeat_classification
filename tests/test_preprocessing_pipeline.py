import os
import shutil
from unittest import TestCase

from src.preprocessing_pipeline import preprocess_physionet_file
from tests.test_signal_utils import TEST_DIR_PATH

TEST_DESTINATION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")


class TestPreprocessingPipeline(TestCase):
    def setUp(self):
        if os.path.exists(TEST_DESTINATION_DIR):
            shutil.rmtree(TEST_DESTINATION_DIR)

    def test_preprocess_data(self):
        os.mkdir(TEST_DESTINATION_DIR)
        for filename in os.listdir(TEST_DIR_PATH):
            preprocess_physionet_file(source_dir=TEST_DIR_PATH, destination_dir=TEST_DESTINATION_DIR, filename=filename)
        self.assertEqual(len(os.listdir(TEST_DESTINATION_DIR)), 102)
