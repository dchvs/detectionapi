#!/usr/bin/env python3

import os
import unittest

import cv2

from detectionapi.detection.detection import Detection


example_input_image = os.path.dirname(
    __file__) + '/../data/example_input_image.jpeg'


class DetectionTests(unittest.TestCase):
    def setUp(self):
        self.this_detection_model = Detection()
        self.assertIsInstance(self.this_detection_model.net, cv2.dnn_Net)

    def test_preprocess(self):
        image = cv2.imread(example_input_image)
        tensor = self.this_detection_model.preprocess(image)
