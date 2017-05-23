.. image:: docs/_static/mycelyso-banner.png

mycelyso Readme
===============

.. image:: https://img.shields.io/pypi/v/mycelyso.svg
   :target: https://pypi.python.org/pypi/mycelyso

.. image:: https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat
   :target: https://mycelyso.readthedocs.io/en/latest/

.. image:: https://travis-ci.org/modsim/mycelyso.svg?branch=master
   :target: https://travis-ci.org/modsim/mycelyso

.. image:: https://img.shields.io/pypi/l/mycelyso.svg
   :target: https://opensource.org/licenses/BSD-2-Clause

.. image:: https://zenodo.org/badge/doi/10.5281/zenodo.376281.svg
   :target: https://dx.doi.org/10.5281/zenodo.376281

.. image:: https://zenodo.org/badge/doi/coming/soon.svg
   :target: https://dx.doi.org/

mycelyso Inspector
------------------
Want to quickly get a grasp wat results can be generated with *mycelyso*?

.. image:: https://modsim.github.io/mycelyso/screenshot.png
   :target: https://modsim.github.io/mycelyso/demo/static/

`Take a look at the static demo page of mycelyso Inspector with the example dataset <https://modsim.github.io/mycelyso/demo/static/>`_.

Publication
-----------
The accompanying publication is currently in submission/review, its citation will be added here once it has been published.

Example Datasets
----------------
You can find an example dataset deposited at zenodo `DOI: 10.5281/zenodo.376281 <https://dx.doi.org/10.5281/zenodo.376281>`_.

Documentation
-------------
Documentation can be built using sphinx, but is available online as well at `Read the Docs <https://mycelyso.readthedocs.io/en/latest/>`_.

License
-------
*mycelyso* is free/libre open source software under the 2-clause BSD License. See :doc:`license`

Prerequisites
-------------
*mycelyso* needs Python 3, if you are running Windows, we suggest a ready to use Python environment like WinPython or conda.

Ways to install mycelyso
------------------------

*mycelyso* is split into two packages, the analysis command line tool *mycelyso* and *mycelyso-inspector* which will serve the frontend.

Once the paper is accepted, a release version will be pushed to PyPI, easing the installation further.

To install *mycelyso-inspector* with the current github Version of *mycelyso*, run the following command:

.. code-block:: bash

    > pip install --user https://github.com/modsim/mycelyso/archive/master.zip mycelyso-inspector

Using mycelyso
--------------

*mycelyso* is packaged as a Python module, to run it, use the following syntax:

.. code-block:: bash

   > python -m mycelyso

Which will produce the help screen:

.. code-block:: none

   mycelyso INFO
     MYCElium   anaLYsis __ SOftware
     ___   __ _________ / /_ _____ ___         Developed  2015 - 2017 by
    /  ' \/ // / __/ -_) / // (_-</ _ \ __
   /_/_/_/\_, /\__/\__/_/\_, /___/\___/'  \.   Christian   C.  Sachs  at
         /___/          /___/              |
               \    `           __     ,''''   Modeling&Simulation Group
                \    `----._ _,'  `'  _/
                 ---'       ''      `-'        Research  Center  Juelich

   If you use this software in a publication, please cite our paper:

   Sachs CC, Koepff J, Wiechert W, Grünberger A, Nöh K (2017)
   mycelyso: Analysis of Streptomyces mycelium live cell imaging data
   Submitted.

   usage: __main__.py [-h] [-m MODULES] [-n PROCESSES] [--prompt]
                      [-tp TIMEPOINTS] [-mp POSITIONS] [-t TUNABLE]
                      [--tunables-show] [--tunables-load TUNABLES_LOAD]
                      [--tunables-save TUNABLES_SAVE] [--meta META] [--box]
                      [--cw CROP_WIDTH] [--ch CROP_HEIGHT] [--si]
                      [--output OUTPUT]
                      input

   positional arguments:
     input                 input file

   optional arguments:
     -h, --help            show this help message and exit
     -m MODULES, --module MODULES
     -n PROCESSES, --processes PROCESSES
     --prompt, --prompt
     -tp TIMEPOINTS, --timepoints TIMEPOINTS
     -mp POSITIONS, --positions POSITIONS
     -t TUNABLE, --tunable TUNABLE
     --tunables-show
     --tunables-load TUNABLES_LOAD
     --tunables-save TUNABLES_SAVE
     --meta META, --meta META
     --box, --detect-box-structure
     --cw CROP_WIDTH, --crop-width CROP_WIDTH
     --ch CROP_HEIGHT, --crop-height CROP_HEIGHT
     --si, --store-image
     --output OUTPUT, --output OUTPUT

To run an analysis, just pass the appropriate filename as a parameter. The desired timepoints can be selected via the
`--timepoints` switch, and if the file contains multiple positions, they can be selected with `--positions`.
Supported file formats are TIFF, OME-TIFF, Nikon ND2 and Zeiss CZI.

To analyze the example dataset, run:
(`--detect-box-structure` is used, as the spores were grown in rectangular growth chambers, which are to be detected.
Otherwise, the software will use the whole image, or cropping values as set via `--cw`/`--ch`.

.. code-block:: bash

   > python -m mycelyso S_lividans_TK24_Complex_Medium_nd046_138.ome.tiff --detect-box-structure

*mycelyso* stores all data in HDF5 files. You can start *mycelyso-inspector* as a helper to take a look at the results:

.. code-block:: bash

   > python -m mycelyso_inspector

WARNING: *mycelyso_inspector* will serve all HDF5 (`.h5`) files found in the current directory via a webserver.
FURTHERMORE, as a research tool, no special focus was laid on security, as such, you are assumed to prevent unauthorized
access to the tool if you choose to use an address accessible by third parties.

Third Party Licenses
--------------------
Note that this software contains the following portions from other authors, under the following licenses (all BSD-flavoured):

molyso/imageio/czifile.py:
    czifile.py by Christoph Gohlke, licensed BSD (see file head).
        Copyright (c) 2013-2015, Christoph Gohlke, 2013-2015, The Regents of the University of California
