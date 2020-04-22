# mocr

[![Build Status](https://github.com/verifid/mocr/workflows/mocr%20ci/badge.svg)](https://github.com/verifid/mocr/actions)
[![pypi](https://img.shields.io/pypi/v/mocr.svg)](https://pypi.python.org/pypi/mocr/)
[![pyversions](https://img.shields.io/pypi/pyversions/mocr.svg)](https://pypi.org/project/mocr)
[![codecov](https://codecov.io/gh/verifid/mocr/branch/master/graph/badge.svg)](https://codecov.io/gh/verifid/mocr)


**Documentation**: <a href="https://mocr.verifid.app" target="_blank">https://mocr.verifid.app</a>

**Source Code**: <a href="https://github.com/verifid/mocr" target="_blank">https://github.com/verifid/mocr</a>


Meaningful Optical Character Recognition from identity cards with Deep Learning.

**mocr** is a library that can be used to detect meaningful optical characters from identity cards. Code base is pure ``Python`` and works with 3.x versions. It has some low level dependencies such as ``Tesseract``. **mocr** uses a pre-trained east detector with OpenCV and applies it's Deep Learning techniques.

It has a pre-trained east detector inside the module and a custom trained model can be given as a parameter.

## Prerequisites

* <a href="https://github.com/tesseract-ocr/tesseract">Tessaract</a> must be installed on your computer before using OCR. Please check <a href="https://github.com/tesseract-ocr/tesseract#installing-tesseract">installation link</a> for details.
* The other dependencies are listed on ``requirements.txt`` and will be installed when you install with pip.
