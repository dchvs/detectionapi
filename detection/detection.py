#!/usr/bin/env python3

import os.path

import cv2
import numpy as np


scale_factor = 1 / 255.0
size_tuple = (640, 640)


class DetectionError(RuntimeError):
    pass


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

    def __init__(self):
        pass


class Detection(Model):
    """ Class that does the Detection model operations.

    ...
    Attributes
    ----------
    net : Net object
        The Yolov5 model object.

    Methods
    -------
    preprocess(image):
        Apply the preprocessing of an input image, i.e., normalizes the image to a size.

    process(tensor):
        Get the predictions out of an input tensor.
    """

    def __init__(self):
        """ Class that does the Detection model operations.
         """
        self.net = cv2.dnn.readNet(
            os.path.dirname(__file__) +
            '/../data/yolov5s.onnx')

    def preprocess(self, image):
        """Apply the preprocessing of an input image, i.e., normalizes the image to a size.

        Parameters
        ----------
        image : data
            The input image to apply the preprocessing.

        Returns
        -------
        tensor : tensor data.
            The 4-dimensional Mat with NCHW dimensions order.

        Raises
        ------
        DetectionError
            If unable to preprocess the input image.
        """
        try:
            col, row, _ = image.shape
            _max = max(col, row)
            normalized_image = np.zeros((_max, _max, 3), np.uint8)
            normalized_image[0:col, 0:row] = image

            tensor = cv2.dnn.blobFromImage(
                normalized_image,
                scalefactor=scale_factor,
                size=size_tuple,
                mean=None,
                swapRB=True,
                crop=None,
                ddepth=None)
        except BaseException:
            raise DetectionError('Unable to preprocess the input image.')

        return tensor

    def process(self, tensor):
        """ Get the predictions out of an input tensor.

        Parameters
        ----------
        tensor : tensor data.
            The input tensor to apply the inference to.

        Returns
        -------
        inference_results : ndarray
            Array with the inference results.

        Raises
        ------
        DetectionError
            If unable to process the input tensor.
        """
        self.net.setInput(tensor)
        predictions = self.net.forward()
        inference_results = predictions[0]

        return inference_results
