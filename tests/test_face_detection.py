#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest
import pytest

from mocr import face_detection

class FaceDetectionTest(unittest.TestCase):

    def test_detect_face_success(self):
        image_path = os.path.join(os.path.dirname(__file__), 'data/sample_de_identity_card.jpg')
        face_image = face_detection.detect_face(image_path)
        self.assertIsNotNone(face_image)

    def test_detect_face_fail(self):
        image_path = os.path.join(os.path.dirname(__file__), 'data/test.png')
        face_image = face_detection.detect_face(image_path)
        self.assertIsNone(face_image)
        image_path = os.path.join(os.path.dirname(__file__), 'data/unavailable.png')
        face_image = face_detection.detect_face(image_path)
        self.assertIsNone(face_image)
