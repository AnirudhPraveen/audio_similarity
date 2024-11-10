"""Tests for the visualization module."""

import pytest
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import json
import tempfile
import torchaudio

from audio_similarity.visualization import AudioSimilaritySearch, BenchmarkResult
from audio_similarity.index_factory import IndexType

class TestAudioSimilaritySearch:
    """Test cases for AudioSimilaritySearch class."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Set up test fixtures."""
        self.test_dir = tmp_path / "test_audio"
        self.test_dir.mkdir()
        
        # Create sample audio files
        self.sample_rate = 16000
        self.duration = 1
        t = torch.linspace(0, self.duration, int(self.sample_rate * self.duration))
        
        # Create different frequencies for unique audio files
        self.test_files = []
        for freq in [440, 880, 1320]:  # A4, A5, E6
            waveform = torch.sin(2 * np.pi * freq * t).unsqueeze(0)
            path = self.test_dir / f"test_{freq}hz.wav"
            torchaudio.save(str(path), waveform, self.sample_rate)
            self.test_files.append(path)
            
        # Initialize searcher
        self.searcher = AudioSimilaritySearch(index_type=IndexType.FLAT)

    def test_initialization(self):
        """Test AudioSimilaritySearch initialization."""
        assert self.searcher.dimension == 768
        assert self.searcher.next_index == 0
        assert isinstance(self.searcher.benchmark_results, list)

    def test_add_audio(self):
        """Test adding single audio file."""
        idx = self.searcher.add_audio(self.test_files[0])
        
        assert idx == 0
        assert len(self.searcher.file_paths) == 1
        assert str(self.test_files[0]) == self.searcher.file_paths[0]

    def test_add_batch(self):
        """Test adding multiple audio files."""
        indices = self.searcher.add_batch(self.test_files)
        
        assert len(indices) == len(self.test_files)
        assert len(self.searcher.file_paths) == len(self.test_files)
        for i, file in enumerate(self.test_files):
            assert str(file) == self.searcher.file_paths[i]

    def test_search(self):
        """Test search functionality."""
        # Add files to index
        self.searcher.add_batch(self.test_files)
        
        # Search using first file as query
        results = self.searcher.search(self.test_files[0], k=2)
        
        assert len(results) == 2
        assert isinstance(results[0], tuple)
        assert len(results[0]) == 2  # (path, distance)
        assert isinstance(results[0][1], float)  # distance should be float

    def test_visualize_search_results(self):
        """Test search results visualization."""
        # Add files and perform search
        self.searcher.add_batch(self.test_files)
        results = self.searcher.search(self.test_files[0], k=2)
        
        # Test visualization without showing
        fig = self.searcher.visualize_search_results(
            self.test_files[0],
            results,
            show=False
        )
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2  # Should have 2 subplots
        
        # Test saving visualization
        with tempfile.NamedTemporaryFile(suffix='.png') as tmp:
            self.searcher.visualize_search_results(
                self.test_files[0],
                results,
                save_path=tmp.name,
                show=False
            )
            assert Path(tmp.name).exists()

    def test_save_load(self, tmp_path):
        """Test saving and loading functionality."""
        # Add some files
        self.searcher.add_batch(self.test_files)
        
        # Save state
        save_dir = tmp_path / "saved_state"
        self.searcher.save(save_dir)
        
        # Check saved files exist
        assert (save_dir / "index.faiss").exists()
        assert (save_dir / "mappings.json").exists()
        
        # Create new searcher and load state
        new_searcher = AudioSimilaritySearch(index_type=IndexType.FLAT)
        new_searcher.load(save_dir)
        
        # Verify loaded state
        assert len(new_searcher.file_paths) == len(self.searcher.file_paths)
        assert new_searcher.next_index == self.searcher.next_index
        
        # Verify search still works
        results = new_searcher.search(self.test_files[0], k=2)
        assert len(results) == 2

    def test_benchmark(self):
        """Test benchmarking functionality."""
        configs = [
            {'type': IndexType.FLAT},
            {'type': IndexType.IVF, 'params': {'nlist': 10}},
            {'type': IndexType.HNSW, 'params': {'M': 16}}
        ]
        
        results = self.searcher.benchmark(
            compare_with=configs,
            num_samples=1000,
            num_queries=10,
            k=5
        )
        
        assert len(results) == len(configs)
        for result in results:
            assert isinstance(result, BenchmarkResult)
            assert result.num_samples == 1000
            assert result.k == 5
            assert 0 <= result.recall <= 1
            assert result.build_time > 0
            assert result.query_time > 0

    @pytest.mark.parametrize("index_type,params", [
        (IndexType.FLAT, {}),
        (IndexType.IVF, {'nlist': 10}),
        (IndexType.HNSW, {'M': 16}),
        (IndexType.PQ, {'M': 8, 'nbits': 8})
    ])
    def test_different_index_types(self, index_type, params):
        """Test with different index types."""
        searcher = AudioSimilaritySearch(
            index_type=index_type,
            index_params=params
        )
        
        # Add files
        searcher.add_batch(self.test_files)
        
        # Test search
        results = searcher.search(self.test_files[0], k=2)
        assert len(results) == 2

    def test_invalid_audio(self):
        """Test handling of invalid audio files."""
        invalid_file = self.test_dir / "invalid.wav"
        invalid_file.write_text("Not an audio file")
        
        with pytest.raises(ValueError):
            self.searcher.add_audio(invalid_file)

    def test_empty_batch(self):
        """Test adding empty batch."""
        indices = self.searcher.add_batch([])
        assert len(indices) == 0

    def test_compute_recall(self):
        """Test recall computation."""
        ground_truth = np.array([[0, 1, 2], [1, 2, 3]])
        results = np.array([[0, 1, 3], [1, 2, 0]])
        recall = self.searcher._compute_recall(ground_truth, results, k=3)
        
        assert isinstance(recall, float)
        assert 0 <= recall <= 1

class TestBenchmarkResult:
    """Test cases for BenchmarkResult class."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample benchmark result."""
        return BenchmarkResult(
            index_type="Flat",
            build_time=1.0,
            query_time=0.1,
            memory_usage=100.0,
            recall=0.95,
            k=5,
            num_samples=1000,
            params={}
        )

    def test_benchmark_result_creation(self, sample_result):
        """Test BenchmarkResult creation and attributes."""
        assert sample_result.index_type == "Flat"
        assert sample_result.build_time == 1.0
        assert sample_result.query_time == 0.1
        assert sample_result.memory_usage == 100.0
        assert sample_result.recall == 0.95
        assert sample_result.k == 5
        assert sample_result.num_samples == 1000
        assert sample_result.params == {}