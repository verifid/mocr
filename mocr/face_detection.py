#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2

def detect_face(image_path):
    """Detect face from given image path.
        Args:
          image_path (str):
            Path to input image on file system.
        Returns:
          image (bytes array):
            Bytes array for detected face image.
    """

    if not os.path.isfile(image_path):
        print('face_detection:detect_face No image found on given image path!')
        return None

    # Create the haar cascade
    cascade_path = os.path.join(os.path.dirname(__file__), 'haarcascades/haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier(cascade_path)
    # Read the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    if len(faces) == 0 or len(faces) > 1:
        print('Identity cards should have a profile picture!')
        return None
    (x, y, w, h) = faces[0]
    cropped_face_image = image[y:y+h, x:x+w]
    return cropped_face_image
