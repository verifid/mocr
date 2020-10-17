mocr
======

.. image:: https://github.com/verifid/mocr/workflows/mocr%20ci/badge.svg
    :target: https://github.com/verifid/mocr/actions

.. image:: https://img.shields.io/pypi/v/mocr.svg
    :target: https://pypi.org/pypi/mocr/

.. image:: https://img.shields.io/pypi/pyversions/mocr.svg
    :target: https://pypi.org/project/mocr

.. image:: https://travis-ci.org/verifid/mocr.svg?branch=master
    :target: https://travis-ci.org/verifid/mocr

.. image:: https://codecov.io/gh/verifid/mocr/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/verifid/mocr


Meaningful Optical Character Recognition from identity cards with Deep Learning.

Introduction
============

**mocr** is a library that can be used to detect meaningful optical characters from identity cards. Code base is pure ``Python`` and
works with 3.x versions. It has some low level dependencies such as ``Tesseract``. **mocr** uses a pre-trained east
detector with OpenCV and applies it's Deep Learning techniques.

It has a pre-trained east detector inside the module and a custom trained model can be given as a parameter.

Prerequisites
=============

* `Tessaract <https://github.com/tesseract-ocr/tesseract>`_ must be installed on your computer before using OCR. Please check `installation link <https://github.com/tesseract-ocr/tesseract#installing-tesseract>`_ for details.
* The other dependencies are listed on ``requirements.txt`` and will be installed when you install with pip.

Installation
============

**From source**

Install module using `pip`::

    $ pip install mocr

Download the latest `mocr` library from: https://github.com/verifid/mocr

Install module using `pip`::

    $ pip install -e .

Extract the source distribution and run::

    $ python setup.py build
    $ python setup.py install

Running Tests
=============

The test suite can be run against a single Python version which requires ``pip install pytest`` and optionally ``pip install pytest-cov`` (these are included if you have installed dependencies from ``requirements.testing.txt``)

To run the unit tests with a single Python version::

    $ py.test -v

to also run code coverage::

    $ py.test -v --cov-report html --cov=mocr

To run the unit tests against a set of Python versions::

    $ tox

Sample Usage
============

* ``text_recognition`` Initiating the ``TextRecognizer`` with identity image and then finding the texts with their frames:

.. code:: python

    import os
    from mocr import TextRecognizer

    image_path = os.path.join('tests', 'data/sample_uk_identity_card.png')
    east_path = os.path.join('mocr', 'model/frozen_east_text_detection.pb')

    text_recognizer = TextRecognizer(image_path, east_path)
    (image, _, _) = text_recognizer.load_image()
    (resized_image, ratio_height, ratio_width, _, _) = text_recognizer.resize_image(image, 320, 320)
    (scores, geometry) = text_recognizer.geometry_score(east_path, resized_image)
    boxes = text_recognizer.boxes(scores, geometry)
    results = text_recognizer.get_results(boxes, image, ratio_height, ratio_width)

    # results: Meaningful texts with bounding boxes

* ``face_detection``:

.. code:: python

    from mocr import face_detection

    image_path = 'YOUR_IDENTITY_IMAGE_PATH'
    face_image = face_detection.detect_face(image_path)
    # face_image is the byte array detected and cropped image from original image

.. code:: python

    from mocr import face_detection

    video_path = 'YOUR_IDENTITY_VIDEO_PATH'
    face_image = face_detection.detect_face_from_video(video_path)
    # face_image is the byte array detected and cropped image from original video

CLI
===

Sample command line usage

* Optical Character Recognition

.. code::

    python -m mocr --image tests/data/sample_uk_identity_card.png --east tests/model/frozen_east_text_detection.pb

* Face detection from image file

.. code::

    python -m mocr --image-face 'tests/data/sample_de_identity_card.jpg'

* Face detection from video file

.. code::

    python -m mocr --video-face 'tests/data/face-demographics-walking.mp4'

Screenshots
-----------

**Before**

|image_before|

**After**

|image_after|

.. |image_before| image:: https://raw.githubusercontent.com/verifid/mocr/master/screenshots/sample_uk_identity_card.png
.. |image_after| image:: https://raw.githubusercontent.com/verifid/mocr/master/screenshots/uk_identity_card_after_detection.png
