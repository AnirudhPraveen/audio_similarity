"""Tests for the utils module."""

import pytest
import torch
import numpy as np
import torchaudio
from pathlib import Path

from audio_similarity.utils import (
    AudioBatch,
    ensure_valid_audio,
    get_audio_statistics,
    load_and_process_audio
)

class TestUtils:
    """Test cases for utility functions."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        self.test_dir = Path(__file__).parent / "test_data"
        self.test_dir.mkdir(exist_ok=True)
        
        # Create test audio files
        self.sample_rate = 16000
        self.duration = 1
        t = torch.linspace(0, self.duration, int(self.sample_rate * self.duration))
        self.waveform = torch.sin(2 * np.pi * 440 * t).unsqueeze(0)
        
        self.test_files = []
        for i in range(3):
            path = self.test_dir / f"test_audio_{i}.wav"
            torchaudio.save(path, self.waveform, self.sample_rate)
            self.test_files.append(path)

        # Create a stereo file
        self.stereo_waveform = torch.cat([self.waveform, self.waveform * 0.5])
        self.stereo_path = self.test_dir / "stereo_audio.wav"
        torchaudio.save(self.stereo_path, self.stereo_waveform, self.sample_rate)

        # Create a different sample rate file
        self.high_sr = 44100
        t_high = torch.linspace(0, self.duration, int(self.high_sr * self.duration))
        self.high_sr_waveform = torch.sin(2 * np.pi * 440 * t_high).unsqueeze(0)
        self.high_sr_path = self.test_dir / "high_sr_audio.wav"
        torchaudio.save(self.high_sr_path, self.high_sr_waveform, self.high_sr)

    def teardown_method(self):
        """Clean up test files after each test."""
        for file in self.test_dir.glob("*.wav"):
            file.unlink()
        self.test_dir.rmdir()

    def test_audio_batch_creation(self):
        """Test AudioBatch creation from files."""
        batch = AudioBatch.from_files(self.test_files)
        
        assert isinstance(batch.waveforms, torch.Tensor)
        assert len(batch.sample_rates) == len(self.test_files)
        assert len(batch.file_paths) == len(self.test_files)
        assert batch.waveforms.shape[0] == len(self.test_files)
        # All waveforms should have the same length
        assert len(set(wf.shape[-1] for wf in batch.waveforms)) == 1

    def test_audio_batch_with_different_lengths(self):
        """Test AudioBatch handling of different length files."""
        # Create files with different lengths
        short_waveform = self.waveform[:, :8000]  # Half duration
        short_path = self.test_dir / "short_audio.wav"
        torchaudio.save(short_path, short_waveform, self.sample_rate)
        
        files = [self.test_files[0], short_path]
        batch = AudioBatch.from_files(files)
        
        # Should pad shorter audio to match longest
        assert batch.waveforms.shape[1] == 1  # mono
        assert batch.waveforms.shape[2] == self.waveform.shape[1]  # longest length

    def test_audio_batch_max_duration(self):
        """Test AudioBatch with max_duration parameter."""
        max_duration = 0.5  # seconds
        batch = AudioBatch.from_files(self.test_files, max_duration=max_duration)
        
        expected_samples = int(max_duration * self.sample_rate)
        assert batch.waveforms.shape[-1] == expected_samples

    def test_audio_batch_target_sample_rate(self):
        """Test AudioBatch with sample rate conversion."""
        target_sr = 8000
        batch = AudioBatch.from_files(
            [self.high_sr_path],
            target_sample_rate=target_sr
        )
        
        assert batch.sample_rates[0] == target_sr

    def test_ensure_valid_audio(self):
        """Test audio file validation."""
        # Test valid file
        assert ensure_valid_audio(self.test_files[0])
        
        # Test nonexistent file
        assert not ensure_valid_audio("nonexistent.wav")
        
        # Test duration checks
        assert ensure_valid_audio(self.test_files[0], min_duration=0.5)
        assert not ensure_valid_audio(self.test_files[0], min_duration=2.0)
        assert ensure_valid_audio(self.test_files[0], max_duration=2.0)
        assert not ensure_valid_audio(self.test_files[0], max_duration=0.5)

    def test_load_and_process_audio(self):
        """Test audio loading and processing."""
        # Test mono file
        waveform, sr = load_and_process_audio(self.test_files[0])
        assert waveform.shape[0] == 1  # mono
        assert sr == self.sample_rate
        
        # Test stereo file
        waveform, sr = load_and_process_audio(self.stereo_path)
        assert waveform.shape[0] == 1  # converted to mono
        assert sr == self.sample_rate
        
        # Test sample rate conversion
        waveform, sr = load_and_process_audio(
            self.high_sr_path,
            target_sample_rate=8000
        )
        assert sr == 8000
        
        # Test max duration
        waveform, sr = load_and_process_audio(
            self.test_files[0],
            max_duration=0.5
        )
        assert waveform.shape[-1] == int(0.5 * sr)

    def test_get_audio_statistics(self):
        """Test audio statistics calculation."""
        stats = get_audio_statistics(self.test_files)
        
        assert isinstance(stats, dict)
        assert all(key in stats for key in [
            "mean_duration",
            "std_duration",
            "total_duration",
            "num_files",
            "min_duration",
            "max_duration"
        ])
        
        assert stats["num_files"] == len(self.test_files)
        assert abs(stats["mean_duration"] - self.duration) < 0.1
        assert stats["std_duration"] == 0  # All files have same duration
        assert stats["total_duration"] == self.duration * len(self.test_files)
        assert stats["min_duration"] == stats["max_duration"]

    def test_get_audio_statistics_different_durations(self):
        """Test statistics with files of different durations."""
        # Create a shorter file
        short_duration = 0.5
        t_short = torch.linspace(0, short_duration, int(self.sample_rate * short_duration))
        short_waveform = torch.sin(2 * np.pi * 440 * t_short).unsqueeze(0)
        short_path = self.test_dir / "short_audio.wav"
        torchaudio.save(short_path, short_waveform, self.sample_rate)
        
        files = [self.test_files[0], short_path]
        stats = get_audio_statistics(files)
        
        assert stats["mean_duration"] == (self.duration + short_duration) / 2
        assert stats["std_duration"] > 0
        assert stats["min_duration"] == short_duration
        assert stats["max_duration"] == self.duration

    def test_get_audio_statistics_empty(self):
        """Test statistics with no files."""
        stats = get_audio_statistics([])
        
        assert stats["num_files"] == 0
        assert stats["mean_duration"] == 0
        assert stats["total_duration"] == 0
        assert stats["std_duration"] == 0
        assert stats["min_duration"] == 0
        assert stats["max_duration"] == 0

    def test_audio_batch_invalid_files(self):
        """Test AudioBatch handling of invalid files."""
        invalid_path = self.test_dir / "invalid.wav"
        with open(invalid_path, 'w') as f:
            f.write("not an audio file")
            
        # Should skip invalid file and process valid ones
        batch = AudioBatch.from_files([self.test_files[0], invalid_path])
        assert len(batch.file_paths) == 1
        
        # Should raise error if no valid files
        with pytest.raises(RuntimeError):
            AudioBatch.from_files([invalid_path])

    def test_audio_batch_empty_input(self):
        """Test AudioBatch with empty input."""
        with pytest.raises(RuntimeError):
            AudioBatch.from_files([])

    def test_load_and_process_audio_invalid(self):
        """Test load_and_process_audio with invalid input."""
        with pytest.raises(ValueError):
            load_and_process_audio("nonexistent.wav")
            
        invalid_path = self.test_dir / "invalid.wav"
        with open(invalid_path, 'w') as f:
            f.write("not an audio file")
            
        with pytest.raises(ValueError):
            load_and_process_audio(invalid_path)