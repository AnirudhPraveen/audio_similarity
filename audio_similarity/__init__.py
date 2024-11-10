"""
Audio Similarity Search Library.

This library provides functionality for audio similarity search using wav2vec2
embeddings and FAISS indexing. It includes tools for audio processing,
similarity search, and performance visualization.

Modules
-------
audio_processor
    Audio file processing and embedding generation
index_factory
    FAISS index creation and management
visualization
    Tools for visualizing search results and benchmarks
utils
    Utility functions for data handling and processing

Examples
--------
>>> from audio_similarity import AudioSimilaritySearch, IndexType
>>> searcher = AudioSimilaritySearch(index_type=IndexType.FLAT)
>>> searcher.index_directory("path/to/audio/files")
>>> results = searcher.search("query.wav", k=5)
"""

from .audio_processor import AudioProcessor
from .index_factory import IndexFactory, IndexType
from .visualization import AudioSimilaritySearch, BenchmarkResult
from .utils import AudioBatch, ensure_valid_audio
from .version import __version__

__all__ = [
    'AudioProcessor',
    'IndexFactory',
    'IndexType',
    'AudioSimilaritySearch',
    'BenchmarkResult',
    'AudioBatch',
    'ensure_valid_audio',
    '__version__',
]

# Set default logging handler to avoid "No handler found" warnings.
import logging
from logging import NullHandler

logging.getLogger(__name__).addHandler(NullHandler())