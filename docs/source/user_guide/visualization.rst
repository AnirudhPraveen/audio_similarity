============================
Visualization Guide
============================

Search Results Visualization
--------------------------

You can visualize search results using the built-in visualization tools:

.. code-block:: python

    from audio_similarity import AudioSimilaritySearch

    # Initialize
    searcher = AudioSimilaritySearch()

    # Perform search
    results = searcher.search("query.wav", k=5)

    # Visualize results
    searcher.visualize_search_results(
        "query.wav",
        results,
        save_path="results.png"
    )