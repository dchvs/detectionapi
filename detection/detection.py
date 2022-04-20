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

    def postprocess(self, input_image, inference_results):
        """ Apply the inference results in terms of bounding boxes into the original frame.

        Parameters
        ----------
        input_image : data.
            The input image to apply this postprocessing.
        inference_results :  tensor data.
            Array with the inference results.

        Returns
        -------
        output_image : data.
            The output image with the inference applied in terms of bounding boxes.

        Raises
        ------
        DetectionError
            If unable to postprocess the inference results.
        """
        class_ids = []
        confidences = []
        boxes = []

        rows = inference_results.shape[0]

        image_width, image_height, _ = input_image.shape

        x_factor = image_width / 640
        y_factor = image_height / 640

        for r in range(rows):
            row = inference_results[r]
            confidence = row[4]
            if confidence >= 0.4:

                classes_scores = row[5:]
                _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
                class_id = max_indx[1]
                if (classes_scores[class_id] > .25):
                    confidences.append(confidence)

                    class_ids.append(class_id)

                    x, y, w, h = row[0].item(), row[1].item(
                    ), row[2].item(), row[3].item()
                    left = int((x - 0.5 * w) * x_factor)
                    top = int((y - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    box = np.array([left, top, width, height])
                    boxes.append(box)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)
        result_class_ids = []
        result_confidences = []
        result_boxes = []
        for i in indexes:
            result_confidences.append(confidences[i])
            result_class_ids.append(class_ids[i])
            result_boxes.append(boxes[i])

        class_list = []
        with open(os.path.dirname(os.path.abspath(__file__)) + "/../data/classes.txt", "r") as f:
            class_list = [cname.strip() for cname in f.readlines()]

        colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]

        for i in range(len(result_class_ids)):
            box = result_boxes[i]
            class_id = result_class_ids[i]

            color = colors[class_id % len(colors)]

            conf = result_confidences[i]

            cv2.rectangle(input_image, box, color, 2)
            cv2.rectangle(
                input_image, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
            cv2.putText(
                input_image,
                class_list[class_id],
                (box[0] + 5,
                 box[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,
                 0,
                 0))

        return result_class_ids, result_confidences, result_boxes

    def is_detection(self, result_class_ids, result_confidences):
        """ Flag that determines whether there was a detection.

        Parameters
        ----------
        result_class_ids : array.
            Array with the inference results for class IDs.
        result_confidences :  array.
            Array with the inference results for confidences.

        Returns
        -------
        detection_flag : bool.
            The flag with the detection result.

        Raises
        ------
        DetectionError
            If unable to set the flag with the detection result.
        """
        try:
            detection_flag = True if len(result_class_ids) > 0 else False
        except BaseException:
            raise DetectionError(
                'Unable to set the flag with the detection result.')

        return detection_flag
