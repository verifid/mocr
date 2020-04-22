# CLI

CLI helps you to detect profile pictures and texts from images.

## Usage

First you need to install **mocr** either from PyPi or source. Then you can use sample
commands below.

Detecs profile picture and save into same folder with given image.

```console
$ python -m mocr --image-face screenshots/sample_uk_identity_card.png
```

Detects texts on given image and creates a new image found texts marked.

```console
$ python -m mocr --image screenshots/sample_uk_identity_card.png --east tests/model/frozen_east_text_detection.pb
```
