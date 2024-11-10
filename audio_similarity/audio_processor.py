"""
Audio processing and embedding generation using wav2vec2.

This module provides functionality for loading audio files and generating
embeddings using Facebook's wav2vec2 model.

Notes
-----
The module assumes audio files are in a format readable by torchaudio
(e.g., WAV, MP3, etc.)
"""

from pathlib import Path
from typing import Tuple, Optional, Union
import logging

import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from torchaudio.transforms import Resample

logger = logging.getLogger(__name__)

class AudioProcessor:
    """
    Audio processing and embedding generation class.

    This class handles loading audio files, preprocessing them to a standard
    format, and generating embeddings using the wav2vec2 model.

    Parameters
    ----------
    cache_dir : str, optional
        Directory for caching wav2vec2 model files
    device : str, optional
        Device to use for computation ('cuda' or 'cpu')
    target_sample_rate : int, optional
        Sample rate to resample audio to, by default 16000
    model_name : str, optional
        Name of the wav2vec2 model to use, by default 'facebook/wav2vec2-base'

    Attributes
    ----------
    device : torch.device
        Device being used for computation
    processor : Wav2Vec2Processor
        Wav2vec2 processor for audio preprocessing
    model : Wav2Vec2Model
        Wav2vec2 model for generating embeddings
    target_sample_rate : int
        Target sample rate for audio processing

    Examples
    --------
    >>> processor = AudioProcessor()
    >>> waveform, sample_rate = processor.load_audio("audio.wav")
    >>> embedding = processor.get_embedding(waveform)
    >>> print(embedding.shape)
    torch.Size([1, 768])
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        target_sample_rate: int = 16000,
        model_name: str = 'facebook/wav2vec2-base'
    ) -> None:
        self.target_sample_rate = target_sample_rate
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize wav2vec2
        self.processor = Wav2Vec2Processor.from_pretrained(
            model_name,
            cache_dir=self.cache_dir
        )
        self.model = Wav2Vec2Model.from_pretrained(
            model_name,
            cache_dir=self.cache_dir
        ).to(self.device)

        self.model.gradient_checkpointing_enable()
        
        self.model.eval()

    def load_audio(
        self,
        audio_path: Union[str, Path]
    ) -> Tuple[torch.Tensor, int]:
        """
        Load and preprocess an audio file.

        Parameters
        ----------
        audio_path : str or Path
            Path to the audio file

        Returns
        -------
        Tuple[torch.Tensor, int]
            - Preprocessed audio waveform
            - Sample rate

        Raises
        ------
        FileNotFoundError
            If the audio file doesn't exist
        RuntimeError
            If the audio file can't be loaded

        Notes
        -----
        The audio will be:
        - Converted to mono if stereo
        - Resampled to target_sample_rate if necessary
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            waveform, sample_rate = torchaudio.load(str(audio_path))
            
            # Convert to mono if necessary
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if necessary
            if sample_rate != self.target_sample_rate:
                resampler = Resample(sample_rate, self.target_sample_rate)
                waveform = resampler(waveform)
                sample_rate = self.target_sample_rate
            
            return waveform, sample_rate
            
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file: {str(e)}")

    def get_embedding(
        self,
        waveform: torch.Tensor,
        return_numpy: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Generate embedding from audio waveform using wav2vec2.
        Parameters
        ----------
        waveform : torch.Tensor
            Audio waveform tensor
        return_numpy : bool, optional
            If True, returns numpy array, otherwise torch tensor
        Returns
        -------
        Union[torch.Tensor, np.ndarray]
            Audio embedding vector of shape (1, 768)
        """
        # Ensure waveform is in correct shape [batch_size, sequence_length]
        if len(waveform.shape) == 2 and waveform.shape[0] == 1:
            # If [1, sequence_length], squeeze first dimension
            waveform = waveform.squeeze(0)
        elif len(waveform.shape) > 2:
            raise ValueError(f"Unexpected waveform shape: {waveform.shape}")
        
        # Add batch dimension if needed
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
            
        print(f"Waveform shape before processor: {waveform.shape}")  # Debug print
        
        # Prepare input
        inputs = self.processor(
            waveform.numpy(),  # processor expects numpy array
            sampling_rate=self.target_sample_rate,
            return_tensors="pt"
        )
        input_values = inputs.input_values.to(self.device)
        
        print(f"Input values shape after processor: {input_values.shape}")  # Debug print
        
        # Generate embedding
        with torch.no_grad():
            outputs = self.model(input_values)
            # Mean pooling over time dimension
            embedding = outputs.last_hidden_state.mean(dim=1)
        
        if return_numpy:
            return embedding.cpu().numpy()
        return embedding

    def process_file(
        self,
        audio_path: Union[str, Path],
        return_numpy: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Process an audio file and return its embedding.

        Parameters
        ----------
        audio_path : str or Path
            Path to the audio file
        return_numpy : bool, optional
            If True, returns numpy array, otherwise torch tensor

        Returns
        -------
        Union[torch.Tensor, np.ndarray]
            Audio embedding vector

        Examples
        --------
        >>> processor = AudioProcessor()
        >>> embedding = processor.process_file("audio.wav")
        >>> print(embedding.shape)
        (1, 768)
        """
        waveform, _ = self.load_audio(audio_path)
        return self.get_embedding(waveform, return_numpy=return_numpy)