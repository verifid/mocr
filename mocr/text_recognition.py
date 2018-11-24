#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2

class TextRecognizer(object):
    """TextRecognizer can be used as to detect meaningful optical characters from identity cards.
    """

    def __init__(self,
                 image_path,
                 east_path='model/frozen_east_text_detection.pb',
                 min_confidence=0.5,
                 width=320,
                 height=320,
                 padding=0.0):
        """Returns a TextRecognizer instance.
        Args:
          image_path (str):
            Path to input image on file system.
          east_path (str):
            Path to input EAST text detector on file system.
          min_confidence (float):
            Minimum probability required to inspect a region.
          width (int):
            Nearest multiple of 32 for resized width.
          height (int):
            Nearest multiple of 32 for resized height.
          padding (float):
            Amount of padding to add to each border of ROI.
        """

        self.image_path = image_path
        self.east_path = east_path
        self.min_confidence = min_confidence
        self.width = width
        self.height = height
        self.padding = padding

    def load_image(self):
        """Load the input image and grab the image dimensions.
        Returns:
          (original, original_height, original_width): Tuple of image, it's height and width.
        """

        image = cv2.imread(self.image_path)
        original = image.copy()
        (original_height, original_width) = image.shape[:2]
        return (original, original_height, original_width)

    def resize_image(self, width, height):
        """Resize the image and grab the new image dimensions.
        Sets the new width and height and then determine the ratio in change
        for both the width and height.
        Args:
          width (int):
            New width to resize.
          height (int):
            New height to resize.
        Returns:
          (resized_image, original_height, original_width, resized_height, resized_width): Resized image, original & resized image size.
        """

        (original, original_height, original_width) = self.load_image()
        ratio_height = original_height / float(height)
        ratio_width = original_width / float(width)

        resized_image = cv2.resize(original, (width, height))
        (resized_height, resized_width) = resized_image.shape[:2]
        return (resized_image, original_height, original_width, resized_height, resized_width)

    def geometry_score(self, east_path, resized_image):
        """Creates scores and geometry to use in predictions.
        Args:
          east_path (str):
            EAST text detector path.
          resized_image (img):
            Resized image data.
        Returns:
          scores (array):
            Probabilities.
          geometry (array):
            Geometrical data.
        """

        (resized_height, resized_width) = resized_image.shape[:2]
        layer_names = [
          "feature_fusion/Conv_7/Sigmoid",
          "feature_fusion/concat_3"]

        # load the pre-trained EAST text detector
        net = cv2.dnn.readNet(east_path)

        # construct a blob from the image and then perform a forward pass of
        # the model to obtain the two output layer sets
        blob = cv2.dnn.blobFromImage(resized_image, 1.0, (resized_width, resized_height),
          (123.68, 116.78, 103.94), swapRB=True, crop=False)
        net.setInput(blob)
        (scores, geometry) = net.forward(layer_names)
        return (scores, geometry)
