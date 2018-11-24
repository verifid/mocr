#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest
import pytest
from mocr import TextRecognizer

class TextRecognizerTest(unittest.TestCase):

    def setUp(self):
        self._image_path = os.path.join('tests', 'data/sample_uk_identity_card.png')
        self._east_path = os.path.join('mocr', 'model/frozen_east_text_detection.pb')
        self._text_recognizer = TextRecognizer(self._image_path)

    def test_init(self):
        self.assertIsNotNone(self._text_recognizer)
        self.assertEqual(self._text_recognizer.image_path, self._image_path)

    def test_load_image(self):
        (original, original_height, original_width) = self._text_recognizer.load_image()
        self.assertIsNotNone(original)
        self.assertEqual(original_height, 201)
        self.assertEqual(original_width, 312)

    def test_resize_image(self):
        (resized_image, original_height, original_width, resized_height, resized_width) = self._text_recognizer.resize_image(320, 320)
        self.assertIsNotNone(resized_image)
        self.assertEqual(original_height, 201)
        self.assertEqual(original_width, 312)
        self.assertEqual(resized_height, 320)
        self.assertEqual(resized_width, 320)

    def test_geometry_score(self):
        (resized_image, _, _, _, _) = self._text_recognizer.resize_image(320, 320)
        (scores, geometry) = self._text_recognizer.geometry_score(self._east_path, resized_image)
        self.assertIsNotNone(scores)
        self.assertIsNotNone(geometry)

    def test_decode_predictions(self):
        (resized_image, _, _, _, _) = self._text_recognizer.resize_image(320, 320)
        (scores, geometry) = self._text_recognizer.geometry_score(self._east_path, resized_image)
        (rects, confidences) = self._text_recognizer.decode_predictions(scores, geometry)
        self.assertIsNotNone(rects)
        self.assertIsNotNone(confidences)
