#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest
import pytest

from mocr import face_detection


class FaceDetectionTest(unittest.TestCase):
    def test_detect_face_success(self):
        image_path = os.path.join(
            os.path.dirname(__file__), "data/sample_de_identity_card.jpg"
        )
        face_image = face_detection.detect_face(image_path)
        self.assertIsNotNone(face_image)

    def test_detect_face_fails(self):
        image_path = os.path.join(os.path.dirname(__file__), "data/test.png")
        face_image = face_detection.detect_face(image_path)
        self.assertIsNone(face_image)
        image_path = os.path.join(os.path.dirname(__file__), "data/unavailable.png")
        face_image = face_detection.detect_face(image_path)
        self.assertIsNone(face_image)

    def test_detect_face_from_video_success(self):
        video_path = os.path.join(
            os.path.dirname(__file__), "data/face-demographics-walking.mp4"
        )
        face_image = face_detection.detect_face_from_video(video_path)
        self.assertIsNotNone(face_image)

    def test_detect_face_from_video_fails(self):
        video_path = os.path.join(os.path.dirname(__file__), "data/unavailable.mp4")
        face_image = face_detection.detect_face_from_video(video_path)
        self.assertIsNone(face_image)

    def main(self):
        self.test_detect_face_success()
        self.test_detect_face_fails()
        self.test_detect_face_from_video_success()
        self.test_detect_face_from_video_fails()


if __name__ == "__main__":
    face_detection_tests = FaceDetectionTest()
    face_detection_tests.main()
