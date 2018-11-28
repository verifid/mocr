Installation & Testing
----------------------

Installation
============

**From PyPI** ::

    $ pip install mocr

**From source**

Install module using `pip`::

    $ pip install -e .

Download the latest `mocr` library from: https://github.com/verifid/mocr

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

Getting the code
================

The code is hosted at `Github <https://github.com/verifid/mocr>`_.

Check out the latest development version anonymously with::

$ git clone https://github.com/verifid/mocr.git
$ cd mocr
