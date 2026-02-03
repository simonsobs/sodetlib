========
sodetlib
========

This repository contains tools for controlling the Simons Observatory readout
system, and performing initial data analysis for detector characterization.

Installation
------------

Instructions for setting up a SMuRF server can be found on `Confluence`_.

For offline analysis of sodetlib data files, you can also install sodetlib
by cloning this repo and running::

    $ python -m pip install -r requirements.txt
    $ python -m pip install .

.. _`Confluence`: https://simonsobs.atlassian.net/wiki/spaces/PRO/pages/11041372/Smurf+Software+Setup

Documentation
-------------

The sodetlib documentation can be built using Sphinx. Once sodetlib and its
dependencies are installed run::

    $ cd docs/
    $ make html

The documentation is also hosted on `Read the Docs`_.

.. _`Read the Docs`: https://sodetlib.readthedocs.io/en/latest/

Contributing
------------

Contributions are very welcome! Pull requests must be approved by one member
of the simonsobs team before being merged.

Licence
-------

This project is licensed under the BSD 2-Clause License - see the `LICENSE`_
file for details.

.. _`LICENSE`: LICENSE
