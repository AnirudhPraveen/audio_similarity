Quick Start Guide
===============

Installation
-----------

Install using conda (recommended):

.. code-block:: bash

   conda create -n audio_sim python=3.12
   conda activate audio_sim
   conda install -c pytorch -c conda-forge pytorch torchaudio faiss
   pip install audio-similarity

Basic Usage
----------

Here's a simple example to get started:

.. code-block:: python

   from audio_similarity import AudioSimilaritySearch, IndexType

   # Initialize
   searcher = AudioSimilaritySearch(index_type=IndexType.FLAT)

   # Add your audio files
   searcher.add_audio("path/to/audio.wav")

   # Search for similar files
   results = searcher.search("query.wav", k=5)

   # Print results
   for file_path, distance in results:
       print(f"Similar file: {file_path}, distance: {distance}")

Examples
--------

Check out more examples in the :doc:`user_guide/index` section.

Next Steps
---------

* Learn about different :doc:`api/index_factory`
* See visualization options in :doc:`user_guide/visualization`
* Read about benchmarking in :doc:`user_guide/benchmarking`