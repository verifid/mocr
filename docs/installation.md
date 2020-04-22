# Installation & Testing

## Installation

**From PyPI**

    $ pip install mocr

**From source**

Install dependencies using `pip`

```console
$ pip install -r requirements.txt
```

Download the latest `mocr` library from: https://github.com/verifid/mocr

Extract the source distribution and run

```console
$ python setup.py build
$ python setup.py install
```

## Running Tests

The test suite can be run against a single Python version which requires `pip install pytest` and optionally `pip install pytest-cov` (these are included if you have installed dependencies from `requirements.testing.txt`)

To run the unit tests with a single Python version

```console
$ py.test -v
```

to also run code coverage

```console
$ py.test -v --cov-report html --cov=mocr
```

To run the unit tests against a set of Python versions::

```console
$ tox
```

## Getting the code

The code is hosted at [Github](https://github.com/verifid/mocr).

Check out the latest development version anonymously with

```console
$ git clone https://github.com/verifid/mocr.git
$ cd mocr
```
