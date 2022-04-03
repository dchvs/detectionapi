#!/usr/bin/env python3

import os
import unittest

from contextlib import contextmanager
import cv2

from detectionapi.detection.detection import Detection


example_input_image = os.path.dirname(
    __file__) + '/../data/example_input_image.jpeg'


class DetectionTests(unittest.TestCase):
    @contextmanager
    def setUp(self):
        self.tensor = None
        self.image = None

        self.this_detection_model = Detection()
        self.assertIsInstance(self.this_detection_model.net, cv2.dnn_Net)

    def test_preprocess(self):
        self.image = cv2.imread(example_input_image)
        self.tensor = self.this_detection_model.preprocess(self.image)

    def test_process(self):
        self.this_detection_model.process(self.tensor)

        yield
