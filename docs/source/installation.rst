============================
Installation Guide
============================

Prerequisites
------------

Before installing the audio-similarity library, ensure you have:

* Python 3.10 or later
* conda package manager
* (Optional) CUDA-capable GPU for faster processing

Basic Installation
----------------

The recommended way to install is using conda:

.. code-block:: bash

    # Create a new conda environment
    conda create -n audio_sim python=3.10
    conda activate audio_sim

    # Install PyTorch ecosystem
    conda install -c pytorch pytorch torchaudio

    # Install FAISS
    conda install -c conda-forge faiss

    # Install the package
    pip install audio-similarity

Installation from Source
----------------------

For development or the latest features, install from source:

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/yourusername/audio-similarity.git
    cd audio-similarity

    # Create conda environment
    conda create -n audio_sim python=3.10
    conda activate audio_sim

    # Install dependencies
    conda install -c pytorch pytorch torchaudio
    conda install -c conda-forge faiss

    # Install in development mode
    pip install -e .

.. GPU Support
.. ----------

.. For GPU support, install the CUDA version of PyTorch and FAISS:

.. .. code-block:: bash

..     conda install -c pytorch pytorch torchaudio pytorch-cuda
..     conda install -c conda-forge faiss-gpu

Verify Installation
-----------------

You can verify your installation by running:

.. code-block:: python

    from audio_similarity import AudioSimilaritySearch
    
    # Initialize the searcher
    searcher = AudioSimilaritySearch()
    print("Installation successful!")

Troubleshooting
--------------

Common Issues
~~~~~~~~~~~~

1. FAISS Installation Issues
   
   If you encounter issues with FAISS, try:

   .. code-block:: bash

       conda install -c conda-forge faiss-cpu

2. PyTorch/Torchaudio Issues

   For M1/M2 Macs:

   .. code-block:: bash

       pip3 install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu

3. Import Errors

   Make sure you're in the correct conda environment:

   .. code-block:: bash

       conda activate audio_sim

Getting Help
~~~~~~~~~~~

If you encounter any issues:

* Check the `GitHub Issues <https://github.com/AnirudhPraveen/audio-similarity/issues>`_