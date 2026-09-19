Installation
============

System Dependencies
-------------------

PathBench requires the following system packages:

- Python 3.10+
- ``espeak-ng`` shared library
- ``build-essential``, ``cmake``
- ``libfftw3-dev``, ``liblapack-dev``

On Ubuntu 22.04::

    sudo apt-get update -qq
    sudo apt install python3 python3-pip python3-venv \
        build-essential cmake espeak-ng libfftw3-dev liblapack-dev -y

Make Installation
-----------------

The recommended installation route uses the provided Makefile::

    git clone git@github.com:karkirowle/pathbench.git
    cd pathbench/tools && make
    cd ..
    source tools/venv/bin/activate

To change the CUDA version::

    make CUDA_VERSION=12.1

For CPU-only::

    make CUDA_VERSION=

Without sudo access, a containerised environment such as Docker is recommended.

.. note::

   PathBench's Python dependencies are available as versioned package-index
   releases, including ``phonemizer-fork==3.3.2`` and
   ``pyctcdecode==0.5.0``. Installing them does not require GitHub credentials.
   The ``espeak-ng`` shared library remains a separate system dependency; use
   the revision documented in the project README for reproducible IPA output.
