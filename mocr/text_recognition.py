#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import cv2
import numpy as np
import pytesseract

from imutils.object_detection import non_max_suppression
from typing import List, Tuple


class TextRecognizer(object):
    """TextRecognizer can be used as to detect meaningful optical characters from identity cards."""

    def __init__(
        self,
        image_path: str,
        east_path: str,
        min_confidence: float = 0.5,
        width: int = 320,
        height: int = 320,
        padding: float = 0.0,
        lang: str = "eng",
    ):
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
          lang (str):
            Language for tessaract.
        """

        self.image_path = image_path
        self.east_path = east_path
        self.min_confidence = min_confidence
        self.width = width
        self.height = height
        self.padding = padding
        self.lang = lang

    def load_image(self) -> Tuple[bytearray, int, int]:
        """Load the input image and grab the image dimensions.
        Returns:
          (original, original_height, original_width): Tuple of image, it's height and width.
        """

        if not os.path.isfile(self.image_path):
            print(
                "mocr:text_recognition:load_image No image found on given image path!"
            )
            return (None, 0, 0)

        image = cv2.imread(self.image_path)
        original = image.copy()
        (original_height, original_width) = image.shape[:2]
        return (original, original_height, original_width)

    def resize_image(
        self, image: bytes, new_width: int, new_height: int
    ) -> Tuple[bytearray, int, int, int, int]:
        """Resize the image and grab the new image dimensions.
        Sets the new width and height and then determine the ratio in change
        for both the width and height.
        Args:
          image (bytes):
            Loaded image data.
          new_width (int):
            New width to resize.
          new_height (int):
            New height to resize.
        Returns:
          (resized_image, ratio_height, ratio_width, resized_height, resized_width): Resized image and it's specs.
        """

        if image is None:
            print("mocr:text_recognition:resize_image Given image is none!")
            return (None, 0, 0, 0, 0)

        original_height, original_width = image.shape[:2]
        ratio_height = original_height / float(new_height)
        ratio_width = original_width / float(new_width)

        resized_image = cv2.resize(image, (new_width, new_height))
        (resized_height, resized_width) = resized_image.shape[:2]
        return (resized_image, ratio_height, ratio_width, resized_height, resized_width)

    def geometry_score(
        self, east_path: str, resized_image: List[bytes]
    ) -> Tuple[List, List]:
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

        if not os.path.isfile(east_path):
            print(
                "mocr:text_recognition:geometry_score No east detector found on given path!"
            )
            return (None, None)

        if resized_image is None:
            print("mocr:text_recognition:geometry_score Given resized_image is none!")
            return (None, None)

        (resized_height, resized_width) = resized_image.shape[:2]
        layer_names = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

        # load the pre-trained EAST text detector
        net = cv2.dnn.readNet(east_path)

        # construct a blob from the image and then perform a forward pass of
        # the model to obtain the two output layer sets
        blob = cv2.dnn.blobFromImage(
            resized_image,
            1.0,
            (resized_width, resized_height),
            (123.68, 116.78, 103.94),
            swapRB=True,
            crop=False,
        )
        net.setInput(blob)
        (scores, geometry) = net.forward(layer_names)
        return (scores, geometry)

    def decode_predictions(self, scores: List, geometry: List) -> Tuple[List, List]:
        """Grab the number of rows and columns from the scores volume, then
        initialize our set of bounding box rectangles and corresponding
        confidence scores.
        Args:
          scores (array):
            Probabilities.
          geometry (array):
            Geometrical data.
        Returns:
          rects (array):
            Bounding boxes.
          confidences (array):
            Associated confidences.
        """

        if scores is None or geometry is None:
            print(
                "mocr:text_recognition:decode_predictions Given scores or geometry is none!"
            )
            return (None, None)

        (num_rows, num_cols) = scores.shape[2:4]
        rects = []
        confidences = []

        # loop over the number of rows
        for y in range(0, num_rows):
            # extract the scores (probabilities), followed by the
            # geometrical data used to derive potential bounding box
            # coordinates that surround text
            scores_data = scores[0, 0, y]
            xdata0 = geometry[0, 0, y]
            xdata1 = geometry[0, 1, y]
            xdata2 = geometry[0, 2, y]
            xdata3 = geometry[0, 3, y]
            angles_data = geometry[0, 4, y]

            # loop over the number of columns
            for x in range(0, num_cols):
                # if our score does not have sufficient probability,
                # ignore it
                if scores_data[x] < self.min_confidence:
                    continue

                # compute the offset factor as our resulting feature
                # maps will be 4x smaller than the input image
                (offset_x, offset_y) = (x * 4.0, y * 4.0)

                # extract the rotation angle for the prediction and
                # then compute the sin and cosine
                angle = angles_data[x]
                cos = np.cos(angle)
                sin = np.sin(angle)

                # use the geometry volume to derive the width and height
                # of the bounding box
                h = xdata0[x] + xdata2[x]
                w = xdata1[x] + xdata3[x]

                # compute both the starting and ending (x, y)-coordinates
                # for the text prediction bounding box
                end_x = int(offset_x + (cos * xdata1[x]) + (sin * xdata2[x]))
                end_y = int(offset_y - (sin * xdata1[x]) + (cos * xdata2[x]))
                start_x = int(end_x - w)
                start_y = int(end_y - h)

                # add the bounding box coordinates and probability score
                # to our respective lists
                rects.append((start_x, start_y, end_x, end_y))
                confidences.append(scores_data[x])

        # return a tuple of the bounding boxes and associated confidences
        return (rects, confidences)

    def boxes(self, scores: List, geometry: List) -> List:
        """Returns boxes after decoding predictions and then applying
        non-maxima suppression.
        Args:
          scores (array):
            Probabilities.
          geometry (array):
            Geometrical data.
        Returns:
          boxes (array):
            Overlapping bounding boxes.
        """

        if scores is None or geometry is None:
            print("mocr:text_recognition:boxes Given scores or geometry is none!")
            return None

        (rects, confidences) = self.decode_predictions(scores, geometry)
        boxes = non_max_suppression(np.array(rects), probs=confidences)
        return boxes

    def get_results(
        self, boxes: List, image: bytes, ratio_height: float, ratio_width: float
    ) -> List:
        """Returns the list of sorted boxes.
        Args:
          boxes (array):
            Overlapping bounding boxes.
          image (bytes):
            Loaded image data.
          ratio_height (float):
            Resize ratio of height.
          ratio_width (float):
            Resize ratio of width.
        Returns:
          results (array):
            Texts with bounding box coordinates from top to bottom.
        """

        if boxes is None or image is None:
            print("mocr:text_recognition:get_results Given boxes or image is none!")
            return None

        (original_height, original_width) = image.shape[:2]
        # initialize the list of results
        results = []
        # loop over the bounding boxes
        for (start_x, start_y, end_x, end_y) in boxes:
            # scale the bounding box coordinates based on the respective
            # ratios
            start_x = int(start_x * ratio_width)
            start_y = int(start_y * ratio_height)
            end_x = int(end_x * ratio_width)
            end_y = int(end_y * ratio_height)

            # in order to obtain a better OCR of the text we can potentially
            # apply a bit of padding surrounding the bounding box -- here we
            # are computing the deltas in both the x and y directions
            dX = int((end_x - start_x) * self.padding)
            dY = int((end_y - start_y) * self.padding)

            # apply padding to each side of the bounding box, respectively
            start_x = max(0, start_x - dX)
            start_y = max(0, start_y - dY)
            end_x = min(original_width, end_x + (dX * 2))
            end_y = min(original_height, end_y + (dY * 2))

            # extract the actual padded ROI
            roi = image[start_y:end_y, start_x:end_x]

            # in order to apply Tesseract v4 to OCR text we must supply
            # (1) a language, (2) an OEM flag of 4, indicating that the we
            # wish to use the LSTM neural net model for OCR, and finally
            # (3) an OEM value, in this case, 7 which implies that we are
            # treating the ROI as a single line of text
            config = str.format("-l {0} --oem 1 --psm 7", self.lang)
            text = pytesseract.image_to_string(roi, config=config)

            # add the bounding box coordinates and OCR'd text to the list
            # of results
            results.append(((start_x, start_y, end_x, end_y), text))

        # sort the results bounding box coordinates from top to bottom
        results = sorted(results, key=lambda r: r[0][1])
        return results
