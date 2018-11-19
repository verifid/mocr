#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

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
