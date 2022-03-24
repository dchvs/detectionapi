#!/usr/bin/env python3

import unittest

import cv2

from detectionapi.detection.detection import Detection


class DetectionTests(unittest.TestCase):
    def setUp(self):
        self.this_detection_model = Detection()

    def test_make(self):
        self.assertIsInstance(self.this_detection_model.net, cv2.dnn_Net)
