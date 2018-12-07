#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import argparse
import cv2

from mocr import TextRecognizer

def display_image(image, results):
    output = image.copy()
    # loop over the results
    for ((startX, startY, endX, endY), text) in results:
        cv2.rectangle(output, (startX, startY), (endX, endY),
                        (0, 0, 255), 2)
    # show the output image
    cv2.imshow('Text Detection', output)
    cv2.imwrite('screenshots/uk_identity_card_after_detection.png', output)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Meaningful Optical Character Recognition from identity cards with Deep Learning.')
    parser.add_argument('--image', type=str, help='Path to input image on file system.')
    parser.add_argument('--east', type=str, help='Path to input EAST text detector on file system.')
    args = parser.parse_args()

    if len(sys.argv) < 4:
        print('Specify an image path and east path')
        sys.exit(1)

    # Optional bash tab completion support
    try:
        import argcomplete
        argcomplete.autocomplete(parser)
    except ImportError:
        pass

    image_path = sys.argv[2]
    east_path = sys.argv[4]
    text_recognizer = TextRecognizer(image_path, east_path)

    (image, _, _) = text_recognizer.load_image()
    (resized_image, ratio_height, ratio_width, _, _) = text_recognizer.resize_image(image, 320, 320)
    (scores, geometry) = text_recognizer.geometry_score(east_path, resized_image)
    boxes = text_recognizer.boxes(scores, geometry)
    results = text_recognizer.get_results(boxes, image, ratio_height, ratio_width)
    display_image(image, results)
