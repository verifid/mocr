#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest
import pytest
from mocr import TextRecognizer

class TextRecognizerTest(unittest.TestCase):

    def setUp(self):
        self._image_path = os.path.join('tests', 'data/sample_uk_identity_card.png')
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
        (resized_image, original_height, original_width, resized_height, resized_width) = self._text_recognizer.resize_image(150, 150)
        self.assertIsNotNone(resized_image)
        self.assertEqual(original_height, 201)
        self.assertEqual(original_width, 312)
        self.assertEqual(resized_height, 150)
        self.assertEqual(resized_width, 150)
