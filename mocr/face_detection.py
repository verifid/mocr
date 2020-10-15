#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2


def detect_face(image_path: str) -> bytearray:
    """Detect face from given image path.
    Args:
      image_path (str):
        Path to input image on file system.
    Returns:
      image (bytes array):
        Bytes array for detected face image.
    """

    if not os.path.isfile(image_path):
        print("mocr:face_detection:detect_face No image found on given image path!")
        return None

    # Create the haar cascade
    cascade_path = os.path.join(
        os.path.dirname(__file__), "haarcascades/haarcascade_frontalface_default.xml"
    )
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
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    if len(faces) == 0 or len(faces) > 1:
        print(
            "mocr:face_detection:detect_face Identity cards should have a profile picture!"
        )
        return None
    (x, y, w, h) = faces[0]
    cropped_face_image = image[y : y + h, x : x + w]
    return cropped_face_image


def detect_face_from_video(video_path: str) -> bytearray:
    """Detect face from given video path.
    Args:
      video_path (str):
        Path to input video on file system.
    Returns:
      image (bytes array):
        Bytes array for detected face image.
    """

    if not os.path.isfile(video_path):
        print(
            "mocr:face_detection:detect_face_from_video No video found on given video path!"
        )
        return None

    # Create the haar cascade
    cascade_path = os.path.join(
        os.path.dirname(__file__), "haarcascades/haarcascade_frontalface_default.xml"
    )
    face_cascade = cv2.CascadeClassifier(cascade_path)

    video_capture = cv2.VideoCapture(video_path)
    faces = []
    cropped_face_image = None
    frame = None
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            break

        if len(faces) == 1:
            break

    if len(faces) == 0 or len(faces) > 1:
        print(
            "mocr:face_detection:detect_face_from_video Video should does not contain a face!"
        )
        return None

    (x, y, w, h) = faces[0]
    cropped_face_image = frame[y : y + h, x : x + w]

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

    return cropped_face_image
