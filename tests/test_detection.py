#!/usr/bin/env python3

import os
import unittest

import cv2

from detectionapi.detection.detection import Detection


example_input_image = os.path.dirname(
    __file__) + '/../data/example_input_image.jpeg'

example_input_image_confidences = 3


class DetectionTests(unittest.TestCase):
    def setUp(self):
        self.tensor = None
        self.image = None

        self.this_detection_model = Detection()
        self.assertIsInstance(self.this_detection_model.net, cv2.dnn_Net)

        self.image = cv2.imread(example_input_image)

    def test_1_preprocess(self):
        self.__class__.tensor = self.this_detection_model.preprocess(
            self.image)

    def test_2_process(self):
        self.__class__.inference_results = self.this_detection_model.process(
            self.__class__.tensor)

    def test_3_postprocess(self):
        self.__class__.class_ids, self.__class__.confidences, boxes = self.this_detection_model.postprocess(
            self.image, self.__class__.inference_results)

        self.assertEqual(
            example_input_image_confidences, len(
                self.__class__.confidences))

    def test_4_is_detection(self):
        is_detection = self.this_detection_model.is_detection(
            self.__class__.class_ids, self.__class__.confidences)
        self.assertTrue(is_detection)
