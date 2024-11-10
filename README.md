# Audio Similarity Search

A Python library for audio similarity search using wav2vec2 embeddings and FAISS indexing. This library provides efficient audio similarity search with support for multiple index types and built-in visualization tools.

## Features

- 🎵 Audio similarity search using wav2vec2 embeddings
- 🚀 Multiple FAISS index types (Flat, IVF, HNSW, PQ)
- 📊 Built-in visualization tools
- 📈 Performance benchmarking
- 🔄 Batch processing support
- 💾 Save and load indices

## Installation

### Prerequisites

- Python 3.10 or later
- conda package manager

### For M1/M2 Mac Users

```bash
# Create conda environment
conda create -n audio_sim python=3.10
conda activate audio_sim

# Install PyTorch ecosystem
pip3 install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu

# Install FAISS
conda install -c conda-forge faiss

# Install the package
pip install audio-similarity
```

### For Other Platforms

```bash
# Create conda environment
conda create -n audio_sim python=3.12
conda activate audio_sim

# Install dependencies
conda install -c pytorch pytorch torchaudio faiss-cpu

# Install the package
pip install audio-similarity
```

### Development Installation

```bash
# Clone the repository
git clone https://github.com/AnirudhPraveen/audio_similarity.git
cd audio-similarity

# Create conda environment
conda create -n audio_sim python=3.12
conda activate audio_sim

# Install dependencies
conda install -c pytorch pytorch torchaudio
conda install -c conda-forge faiss

# Install in development mode
pip install -e .
```

## Quick Start

```python
from audio_similarity import AudioSimilaritySearch, IndexType

# Initialize
searcher = AudioSimilaritySearch(index_type=IndexType.FLAT)

# Add audio files
searcher.add_audio("path/to/audio1.wav")
searcher.add_audio("path/to/audio2.wav")

# Search for similar files
results = searcher.search("path/to/query.wav", k=5)

# Print results
for file_path, distance in results:
    print(f"Similar file: {file_path}, distance: {distance}")

# Visualize results
searcher.visualize_search_results(
    "path/to/query.wav",
    results,
    save_path="results.png"
)
```

## Advanced Usage

### Batch Processing

```python
from pathlib import Path

# Get all audio files in a directory
audio_dir = Path("path/to/audio/files")
audio_files = list(audio_dir.glob("*.wav"))

# Add files in batch
searcher.add_batch(audio_files)
```

### Different Index Types

```python
# Exact search (slower but accurate)
searcher = AudioSimilaritySearch(index_type=IndexType.FLAT)

# Approximate search (faster)
searcher = AudioSimilaritySearch(
    index_type=IndexType.IVF,
    index_params={'nlist': 100}
)

# Graph-based search (memory intensive but fast)
searcher = AudioSimilaritySearch(
    index_type=IndexType.HNSW,
    index_params={'M': 16}
)
```

### Benchmarking

```python
# Compare different index types
configs = [
    {'type': IndexType.FLAT},
    {'type': IndexType.IVF, 'params': {'nlist': 100}},
    {'type': IndexType.HNSW, 'params': {'M': 16}},
]

results = searcher.benchmark(
    compare_with=configs,
    num_samples=1000,
    num_queries=100,
    k=5
)

# Visualize benchmark results
searcher.visualize_benchmarks()
```

<!-- ## Documentation

Full documentation is available at [Read the Docs](https://audio-similarity.readthedocs.io/). -->

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a new branch: `git checkout -b feature-name`
3. Make your changes and commit: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a Pull Request

## Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=audio_similarity tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{audio_similarity2024,
  author = {Anirudh Praveen},
  title = {Audio Similarity Search},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/AnirudhPraveen/audio_similarity}
}
```

## Acknowledgments

- Facebook AI Research for wav2vec2
- Facebook Research for FAISS
- PyTorch team for torch and torchaudio

## Contact

- GitHub Issues: [Project Issues](https://github.com/AnirudhPraveen/audio_similarity/issues)
- Email: anirudhpraveen2000@example.com