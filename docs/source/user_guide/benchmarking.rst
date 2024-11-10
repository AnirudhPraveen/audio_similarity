============================
Benchmarking
============================

Using the Benchmark Tools
------------------------

The library provides tools to benchmark different FAISS index types:

.. code-block:: python

    from audio_similarity import AudioSimilaritySearch, IndexType

    # Initialize searcher
    searcher = AudioSimilaritySearch()

    # Define configurations to compare
    configs = [
        {'type': IndexType.FLAT},
        {'type': IndexType.IVF, 'params': {'nlist': 100}},
        {'type': IndexType.HNSW, 'params': {'M': 16}},
    ]

    # Run benchmark
    results = searcher.benchmark(
        compare_with=configs,
        num_samples=1000,
        num_queries=100,
        k=5
    )

    # Visualize results
    searcher.visualize_benchmarks()