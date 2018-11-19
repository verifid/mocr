#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import pytest
from mocr import TextRecognizer

class TextRecognizerTest(unittest.TestCase):

    def setUp(self):
        self._text_recognizer = TextRecognizer('data/sample_uk_identity_card.png')

    def test_init(self):
        self.assertIsNotNone(self._text_recognizer)
        self.assertEqual(self._text_recognizer.image_path, 'data/sample_uk_identity_card.png')
