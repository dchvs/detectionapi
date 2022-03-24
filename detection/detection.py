#!/usr/bin/env python3

import os.path

import cv2


class Model:
    """
    Class that abstracts the Machine Learning models.

    Attributes
    ----------

    Methods
    -------

    Raises
    ------
    """
    pass


class Detection(Model):
    """
    Class that does the Detection model operations.

    Attributes
    ----------
    net : Net object
        The Yolov5 model object.

    Methods
    -------

    Raises
    ------
    """

    def __init__(self):
        """
         Parameters
         ----------
         """
        self.net = cv2.dnn.readNet(
            os.path.dirname(__file__) +
            '/../data/yolov5s.onnx')
