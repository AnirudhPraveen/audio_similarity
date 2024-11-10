"""Test suite for the audio_similarity package."""

import os
import pytest
import torch
import numpy as np
from pathlib import Path

# Create a fixture for test audio files
@pytest.fixture
def test_audio_dir():
    """Create a temporary directory with test audio files."""
    return Path(__file__).parent / "test_data"

@pytest.fixture
def sample_waveform():
    """Create a sample audio waveform for testing."""
    # Create a simple sine wave
    sample_rate = 16000
    duration = 1  # seconds
    t = torch.linspace(0, duration, int(sample_rate * duration))
    waveform = torch.sin(2 * np.pi * 440 * t).unsqueeze(0)  # 440 Hz sine wave
    return waveform, sample_rate

@pytest.fixture
def sample_embedding():
    """Create a sample embedding for testing."""
    # Create random embedding of correct size (wav2vec2 dimension is 768)
    return torch.randn(1, 768)