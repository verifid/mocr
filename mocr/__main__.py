#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import cv2

from mocr import TextRecognizer, face_detection


def display_image(image, results, file_name):
    output = image.copy()
    # loop over the results
    for ((startX, startY, endX, endY), text) in results:
        cv2.rectangle(output, (startX, startY), (endX, endY), (0, 0, 255), 2)
    # show the output image
    cv2.imshow("Text Detection", output)
    cv2.imwrite("screenshots/processed_" + file_name, output)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Meaningful Optical Character Recognition from identity cards with Deep Learning."
    )
    parser.add_argument("--image", type=str, help="Path to input image on file system.")
    parser.add_argument(
        "--east", type=str, help="Path to input EAST text detector on file system."
    )
    parser.add_argument(
        "--image-face", type=str, help="Path to input image on file system."
    )
    parser.add_argument(
        "--video-face", type=str, help="Path to input video on file system."
    )
    args = parser.parse_args()

    # Optional bash tab completion support
    try:
        import argcomplete

        argcomplete.autocomplete(parser)
    except ImportError:
        pass

    if sys.argv[1] == "--image-face":
        if len(sys.argv) < 3:
            print("Specify an image path")
            sys.exit(1)

        image_path = sys.argv[2]
        file_name = os.path.basename(image_path)
        face = face_detection.detect_face(image_path)
        cv2.imshow("Found profile", face)
        cv2.imwrite("screenshots/profile_" + file_name, face)
    elif sys.argv[1] == "--video-face":
        if len(sys.argv) < 3:
            print("Specify an video path")
            sys.exit(1)

        video_path = sys.argv[2]
        base = os.path.basename(video_path)
        file_name = os.path.splitext(base)[0]
        face = face_detection.detect_face_from_video(video_path)
        cv2.imshow("Found profile", face)
        print(file_name)
        cv2.imwrite("screenshots/profile_" + file_name + ".png", face)
    else:
        if len(sys.argv) < 4:
            print("Specify an image path and east path")
            sys.exit(1)

        image_path = sys.argv[2]
        east_path = sys.argv[4]
        text_recognizer = TextRecognizer(image_path, east_path)
        file_name = os.path.basename(image_path)

        (image, _, _) = text_recognizer.load_image()
        (resized_image, ratio_height, ratio_width, _, _) = text_recognizer.resize_image(
            image, 320, 320
        )
        (scores, geometry) = text_recognizer.geometry_score(east_path, resized_image)
        boxes = text_recognizer.boxes(scores, geometry)
        results = text_recognizer.get_results(boxes, image, ratio_height, ratio_width)
        display_image(image, results, file_name)
