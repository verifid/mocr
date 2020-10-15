#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest
import pytest
from mocr import TextRecognizer


class TextRecognizerTest(unittest.TestCase):
    def setUp(self):
        self._image_path = os.path.join("tests", "data/sample_uk_identity_card.png")
        self._east_path = os.path.join("tests", "model/frozen_east_text_detection.pb")
        self._text_recognizer = TextRecognizer(self._image_path, self._east_path)

    def test_init(self):
        self.assertIsNotNone(self._text_recognizer)
        self.assertEqual(self._text_recognizer.image_path, self._image_path)

    def test_load_image(self):
        (original, original_height, original_width) = self._text_recognizer.load_image()
        self.assertIsNotNone(original)
        self.assertEqual(original_height, 201)
        self.assertEqual(original_width, 312)

    def test_resize_image(self):
        (image, _, _) = self._text_recognizer.load_image()
        (
            resized_image,
            ratio_height,
            ratio_width,
            resized_height,
            resized_width,
        ) = self._text_recognizer.resize_image(image, 320, 320)
        self.assertIsNotNone(resized_image)
        self.assertEqual(ratio_height, 0.628125)
        self.assertEqual(ratio_width, 0.975)
        self.assertEqual(resized_height, 320)
        self.assertEqual(resized_width, 320)

    def test_geometry_score(self):
        (image, _, _) = self._text_recognizer.load_image()
        (resized_image, _, _, _, _) = self._text_recognizer.resize_image(
            image, 320, 320
        )
        (scores, geometry) = self._text_recognizer.geometry_score(
            self._east_path, resized_image
        )
        self.assertIsNotNone(scores)
        self.assertIsNotNone(geometry)

    def test_geometry_score_fail(self):
        image_path = os.path.join("tests", "data/sample_uk_identity_card.png")
        east_path = os.path.join("tests", "model/unavailable.pb")
        text_recognizer = TextRecognizer(image_path, east_path)
        (image, _, _) = text_recognizer.load_image()
        (resized_image, _, _, _, _) = text_recognizer.resize_image(image, 320, 320)
        (scores, geometry) = text_recognizer.geometry_score(east_path, resized_image)
        self.assertIsNone(scores)
        self.assertIsNone(geometry)

    def test_decode_predictions(self):
        (image, _, _) = self._text_recognizer.load_image()
        (resized_image, _, _, _, _) = self._text_recognizer.resize_image(
            image, 320, 320
        )
        (scores, geometry) = self._text_recognizer.geometry_score(
            self._east_path, resized_image
        )
        (rects, confidences) = self._text_recognizer.decode_predictions(
            scores, geometry
        )
        self.assertIsNotNone(rects)
        self.assertIsNotNone(confidences)

    def test_boxes(self):
        (image, _, _) = self._text_recognizer.load_image()
        (resized_image, _, _, _, _) = self._text_recognizer.resize_image(
            image, 320, 320
        )
        (scores, geometry) = self._text_recognizer.geometry_score(
            self._east_path, resized_image
        )
        boxes = self._text_recognizer.boxes(scores, geometry)
        self.assertIsNotNone(boxes)

    def test_get_results(self):
        (image, _, _) = self._text_recognizer.load_image()
        (
            resized_image,
            ratio_height,
            ratio_width,
            _,
            _,
        ) = self._text_recognizer.resize_image(image, 320, 320)
        (scores, geometry) = self._text_recognizer.geometry_score(
            self._east_path, resized_image
        )
        boxes = self._text_recognizer.boxes(scores, geometry)
        results = self._text_recognizer.get_results(
            boxes, image, ratio_height, ratio_width
        )
        self.assertIsNotNone(results)

    def test_de_get_results(self):
        image_path = os.path.join("tests", "data/sample_de_identity_card.jpg")
        text_recognizer = TextRecognizer(image_path, self._east_path, lang="deu")
        (image, _, _) = text_recognizer.load_image()
        (resized_image, ratio_height, ratio_width, _, _) = text_recognizer.resize_image(
            image, 320, 320
        )
        (scores, geometry) = text_recognizer.geometry_score(
            self._east_path, resized_image
        )
        boxes = text_recognizer.boxes(scores, geometry)
        results = text_recognizer.get_results(boxes, image, ratio_height, ratio_width)
        self.assertIsNotNone(results)

    def test_get_results_fail(self):
        image_path = os.path.join("tests", "data/unavailable.png")
        east_path = os.path.join("tests", "model/frozen_east_text_detection.pb")
        text_recognizer = TextRecognizer(image_path, east_path)
        (image, _, _) = text_recognizer.load_image()
        self.assertIsNone(image)
        (resized_image, ratio_height, ratio_width, _, _) = text_recognizer.resize_image(
            image, 320, 320
        )
        self.assertIsNone(resized_image)
        (scores, geometry) = text_recognizer.geometry_score(east_path, resized_image)
        self.assertIsNone(scores)
        self.assertIsNone(geometry)
        boxes = text_recognizer.boxes(scores, geometry)
        self.assertIsNone(boxes)
        results = text_recognizer.get_results(boxes, image, ratio_height, ratio_width)
        self.assertIsNone(results)

    def main(self):
        self.setUp()
        self.test_init()
        self.test_load_image()
        self.test_resize_image()
        self.test_geometry_score()
        self.test_geometry_score_fail()
        self.test_decode_predictions()
        self.test_boxes()
        self.test_get_results()
        self.test_de_get_results()
        self.test_get_results_fail()


if __name__ == "__main__":
    text_recognizer_tests = TextRecognizerTest()
    text_recognizer_tests.main()
