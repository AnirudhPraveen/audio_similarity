"""Tests for the audio_processor module."""

import pytest
import torch
import numpy as np
import torchaudio
import transformers
from pathlib import Path

from audio_similarity.audio_processor import AudioProcessor

@pytest.fixture
def sample_waveform():
    """Create a sample audio waveform for testing."""
    sample_rate = 16000
    duration = 1  # seconds
    t = torch.linspace(0, duration, int(sample_rate * duration))
    waveform = torch.sin(2 * np.pi * 440 * t).unsqueeze(0)  # 440 Hz sine wave
    return waveform, sample_rate

@pytest.fixture
def test_audio_dir(tmp_path):
    """Create a temporary directory for test audio files."""
    test_dir = tmp_path / "test_audio"
    test_dir.mkdir()
    return test_dir

@pytest.fixture
def test_audio_file(test_audio_dir, sample_waveform):
    """Create a test audio file."""
    waveform, sample_rate = sample_waveform
    file_path = test_audio_dir / "test_audio.wav"
    torchaudio.save(str(file_path), waveform, sample_rate)
    return file_path

class TestAudioProcessor:
    """Test cases for AudioProcessor class."""

    @pytest.fixture(autouse=True)
    def setup(self, test_audio_dir, sample_waveform):
        """Set up test fixtures."""
        self.processor = AudioProcessor()
        self.waveform, self.sample_rate = sample_waveform
        self.test_audio_path = test_audio_dir / "test_audio.wav"
        
        # Save test audio file
        torchaudio.save(str(self.test_audio_path), self.waveform, self.sample_rate)

    def test_initialization(self):
        """Test AudioProcessor initialization."""
        assert self.processor.target_sample_rate == 16000
        assert isinstance(self.processor.processor, transformers.Wav2Vec2Processor)
        assert isinstance(self.processor.model, transformers.Wav2Vec2Model)

    def test_load_audio(self):
        """Test audio loading functionality."""
        waveform, sample_rate = self.processor.load_audio(self.test_audio_path)
        
        assert isinstance(waveform, torch.Tensor)
        assert waveform.dim() == 2  # [channels, samples]
        assert sample_rate == self.processor.target_sample_rate

    def test_load_audio_nonexistent_file(self):
        """Test loading nonexistent audio file raises error."""
        with pytest.raises(FileNotFoundError):
            self.processor.load_audio("nonexistent.wav")

    def test_get_embedding(self, sample_waveform):
        """Test embedding generation."""
        waveform, _ = sample_waveform
        embedding = self.processor.get_embedding(waveform)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1, 768)  # wav2vec2-base embedding dimension

    def test_get_embedding_torch_output(self, sample_waveform):
        """Test embedding generation with torch output."""
        waveform, _ = sample_waveform
        embedding = self.processor.get_embedding(waveform, return_numpy=False)
        
        assert isinstance(embedding, torch.Tensor)
        assert embedding.shape == (1, 768)

    def test_process_file(self):
        """Test complete file processing."""
        embedding = self.processor.process_file(self.test_audio_path)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1, 768)

    def test_batch_processing(self, test_audio_dir):
        """Test processing multiple files."""
        # Create multiple test files
        test_files = []
        for i in range(3):
            path = test_audio_dir / f"test_audio_{i}.wav"
            torchaudio.save(str(path), self.waveform, self.sample_rate)
            test_files.append(path)
        
        embeddings = [self.processor.process_file(str(f)) for f in test_files]
        embeddings = np.vstack(embeddings)
        
        assert embeddings.shape == (3, 768)

    def test_sample_rate_conversion(self, test_audio_dir):
        """Test audio sample rate conversion."""
        # Create test audio with different sample rate
        high_sample_rate = 44100
        duration = 1
        t = torch.linspace(0, duration, int(high_sample_rate * duration))
        waveform = torch.sin(2 * np.pi * 440 * t).unsqueeze(0)
        
        path = test_audio_dir / "test_audio_44100.wav"
        torchaudio.save(str(path), waveform, high_sample_rate)
        
        loaded_waveform, loaded_sr = self.processor.load_audio(path)
        assert loaded_sr == self.processor.target_sample_rate

    def test_stereo_to_mono_conversion(self, test_audio_dir):
        """Test stereo to mono conversion."""
        # Create stereo audio
        stereo_waveform = torch.cat([self.waveform, self.waveform * 0.5])
        
        path = test_audio_dir / "test_audio_stereo.wav"
        torchaudio.save(str(path), stereo_waveform, self.sample_rate)
        
        loaded_waveform, _ = self.processor.load_audio(path)
        assert loaded_waveform.shape[0] == 1  # Check if mono

    @pytest.mark.parametrize("duration", [0.5, 1.0, 2.0])
    def test_different_durations(self, test_audio_dir, duration):
        """Test processing audio files of different durations."""
        t = torch.linspace(0, duration, int(self.sample_rate * duration))
        waveform = torch.sin(2 * np.pi * 440 * t).unsqueeze(0)
        
        path = test_audio_dir / f"test_audio_{duration}s.wav"
        torchaudio.save(str(path), waveform, self.sample_rate)
        
        embedding = self.processor.process_file(path)
        assert embedding.shape == (1, 768)

    def test_invalid_audio_format(self, test_audio_dir):
        """Test handling of invalid audio format."""
        invalid_path = test_audio_dir / "invalid.wav"
        invalid_path.write_text("Not an audio file")
        
        with pytest.raises((RuntimeError, ValueError)):
            self.processor.load_audio(invalid_path)

    def test_empty_audio(self, test_audio_dir):
        """Test handling of empty audio file."""
        empty_waveform = torch.zeros((1, 100))
        empty_path = test_audio_dir / "empty.wav"
        torchaudio.save(str(empty_path), empty_waveform, self.sample_rate)
        
        embedding = self.processor.process_file(empty_path)
        assert embedding.shape == (1, 768)

    def test_model_device(self):
        """Test model device placement."""
        expected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert self.processor.model.device == expected_device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_processing(self):
        """Test processing on GPU if available."""
        embedding = self.processor.process_file(self.test_audio_path)
        assert isinstance(embedding, np.ndarray)  # Should work regardless of device