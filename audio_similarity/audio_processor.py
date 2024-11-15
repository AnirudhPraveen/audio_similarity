"""
Audio processing and embedding generation module.

This module provides functionality for loading audio files and generating
embeddings using either mean pooling (AudioProcessor) or TF-IDF weighting 
(AudioTfidfProcessor) of wav2vec2 features.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from collections import Counter

import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from torchaudio.transforms import Resample

logger = logging.getLogger(__name__)

class AudioProcessor:
    """
    Base audio processing class using wav2vec2 with mean pooling.

    Args:
        cache_dir (Optional[str]): Directory for caching wav2vec2 model files.
        device (Optional[str]): Device to use for computation ('cuda' or 'cpu').
        target_sample_rate (int): Sample rate to resample audio to. Defaults to 16000.
        model_name (str): Name of the wav2vec2 model to use. 
            Defaults to 'facebook/wav2vec2-base'.

    Examples:
        >>> processor = AudioProcessor()
        >>> embedding = processor.process_file("audio.wav")
        >>> print(embedding.shape)
        (1, 768)
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
        
        self.model.eval()

    def load_audio(
        self,
        audio_path: Union[str, Path]
    ) -> Tuple[torch.Tensor, int]:
        """
        Load and preprocess an audio file.

        Args:
            audio_path (Union[str, Path]): Path to the audio file.

        Returns:
            Tuple[torch.Tensor, int]: Tuple containing:
                - Preprocessed audio waveform
                - Sample rate

        Raises:
            FileNotFoundError: If the audio file doesn't exist.
            RuntimeError: If the audio file can't be loaded.
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

        Args:
            waveform (torch.Tensor): Audio waveform tensor.
            return_numpy (bool): If True, returns numpy array, otherwise torch tensor.

        Returns:
            Union[torch.Tensor, np.ndarray]: Audio embedding vector of shape (1, 768).
        """
        if len(waveform.shape) == 2 and waveform.shape[0] == 1:
            waveform = waveform.squeeze(0)
        
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
            
        inputs = self.processor(
            waveform.numpy(),
            sampling_rate=self.target_sample_rate,
            return_tensors="pt"
        )
        input_values = inputs.input_values.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_values)
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

        Args:
            audio_path (Union[str, Path]): Path to the audio file.
            return_numpy (bool): If True, returns numpy array, otherwise torch tensor.

        Returns:
            Union[torch.Tensor, np.ndarray]: Audio embedding vector.
        """
        waveform, _ = self.load_audio(audio_path)
        return self.get_embedding(waveform, return_numpy=return_numpy)


class AudioTfidfProcessor(AudioProcessor):
    """
    Audio processing class using wav2vec2 with TF-IDF weighting.
    
    This class extends AudioProcessor to use TF-IDF weighting instead of mean pooling
    for generating embeddings from wav2vec2 features.

    Args:
        cache_dir (Optional[str]): Directory for caching wav2vec2 model files.
        device (Optional[str]): Device to use for computation ('cuda' or 'cpu').
        target_sample_rate (int): Sample rate to resample audio to. Defaults to 16000.
        model_name (str): Name of the wav2vec2 model to use. 
            Defaults to 'facebook/wav2vec2-base'.
        num_bins (int): Number of bins for discretizing feature values. Defaults to 100.

    Examples:
        >>> processor = AudioTfidfProcessor()
        >>> embedding = processor.process_file("audio.wav")
        >>> print(embedding.shape)
        (1, 768)
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        target_sample_rate: int = 16000,
        model_name: str = 'facebook/wav2vec2-base',
        num_bins: int = 100
    ) -> None:
        super().__init__(cache_dir, device, target_sample_rate, model_name)
        self.num_bins = num_bins
        self.feature_mins = None
        self.feature_maxs = None

    def create_bins(
        self,
        feature_range: Tuple[float, float]
    ) -> np.ndarray:
        """
        Creates evenly spaced bins for feature discretization.

        Args:
            feature_range (Tuple[float, float]): The (min, max) range of feature values.

        Returns:
            np.ndarray: Array of bin edges including leftmost and rightmost edges.
        """
        return np.linspace(feature_range[0], feature_range[1], self.num_bins + 1)

    def featurize_timestep(
        self,
        features: np.ndarray,
        bins: np.ndarray
    ) -> str:
        """
        Converts a single timestep's features into a token string.

        Args:
            features (np.ndarray): 1D array of feature values for one timestep.
            bins (np.ndarray): Array of bin edges for discretization.

        Returns:
            str: Token representing the binned features.
        """
        bin_indices = np.digitize(features, bins)
        tokens = [f"f{i}_b{b}" for i, b in enumerate(bin_indices)]
        return "_".join(tokens)

    def calculate_tf(
        self,
        tokens: List[str]
    ) -> Dict[str, int]:
        """
        Calculates term frequencies for a list of tokens.

        Args:
            tokens (List[str]): List of tokens representing feature patterns.

        Returns:
            Dict[str, int]: Dictionary mapping tokens to their frequency counts.
        """
        return dict(Counter(tokens))

    def calculate_idf(
        self,
        tf_dict: Dict[str, int],
        n_samples: int
    ) -> Dict[str, float]:
        """
        Calculates inverse document frequencies from term frequencies.

        Args:
            tf_dict (Dict[str, int]): Dictionary of term frequencies.
            n_samples (int): Total number of samples (timesteps).

        Returns:
            Dict[str, float]: Dictionary mapping tokens to their IDF scores.
        """
        return {term: np.log(n_samples / count) for term, count in tf_dict.items()}

    def calculate_tfidf(
        self,
        features: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculates TF-IDF scores for audio feature patterns.

        Args:
            features (np.ndarray): 2D array of shape (timesteps, features).

        Returns:
            Dict[str, float]: Dictionary mapping feature patterns to TF-IDF scores.

        Raises:
            ValueError: If features array is not 2D or is empty.
        """
        if len(features.shape) != 2 or features.size == 0:
            raise ValueError(f"Expected non-empty 2D array, got shape {features.shape}")

        # Update or initialize feature statistics
        if self.feature_mins is None:
            self.feature_mins = np.min(features, axis=0)
            self.feature_maxs = np.max(features, axis=0)

        # Create bins for discretization
        bins = self.create_bins((np.min(self.feature_mins), np.max(self.feature_maxs)))

        # Convert each timestep to a token
        tokens = []
        for timestep_features in features:
            token = self.featurize_timestep(timestep_features, bins)
            tokens.append(token)

        # Calculate TF and IDF
        tf_dict = self.calculate_tf(tokens)
        idf_dict = self.calculate_idf(tf_dict, len(tokens))

        # Calculate final TF-IDF scores
        tfidf_dict = {
            term: count * idf_dict[term]
            for term, count in tf_dict.items()
        }

        return tfidf_dict

    def get_embedding(
        self,
        waveform: torch.Tensor,
        return_numpy: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Generate TF-IDF weighted embedding from audio waveform using wav2vec2.

        Args:
            waveform (torch.Tensor): Audio waveform tensor.
            return_numpy (bool): If True, returns numpy array, otherwise torch tensor.

        Returns:
            Union[torch.Tensor, np.ndarray]: Audio embedding vector of shape (1, 768).

        Raises:
            ValueError: If waveform has unexpected shape.
        """
        # Ensure waveform is in correct shape
        if len(waveform.shape) == 2 and waveform.shape[0] == 1:
            waveform = waveform.squeeze(0)
        elif len(waveform.shape) > 2:
            raise ValueError(f"Unexpected waveform shape: {waveform.shape}")
        
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
        
        # Get wav2vec2 features
        inputs = self.processor(
            waveform.numpy(),
            sampling_rate=self.target_sample_rate,
            return_tensors="pt"
        )
        input_values = inputs.input_values.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_values)
            features = outputs.last_hidden_state.cpu().numpy()
        
        # Calculate TF-IDF scores
        tfidf_dict = self.calculate_tfidf(features.reshape(-1, features.shape[-1]))
        
        # Create sparse vector of TF-IDF scores
        unique_patterns = sorted(tfidf_dict.keys())
        tfidf_vector = np.array([tfidf_dict[pattern] for pattern in unique_patterns])
        
        # Project to target dimension (768)
        projection_matrix = np.random.randn(len(tfidf_vector), 768)
        embedding = tfidf_vector.dot(projection_matrix)
        
        # Normalize the embedding
        embedding = embedding / np.linalg.norm(embedding)
        embedding = embedding.reshape(1, -1)
        
        if not return_numpy:
            embedding = torch.from_numpy(embedding).float()
        
        return embedding
    
    def process_file(
        self,
        audio_path: Union[str, Path],
        return_numpy: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Process an audio file and return its embedding.

        Args:
            audio_path (Union[str, Path]): Path to the audio file.
            return_numpy (bool): If True, returns numpy array, otherwise torch tensor.

        Returns:
            Union[torch.Tensor, np.ndarray]: Audio embedding vector.
        """
        waveform, _ = self.load_audio(audio_path)
        return self.get_embedding(waveform, return_numpy=return_numpy)


# Make both classes available when importing from this module
__all__ = ['AudioProcessor', 'AudioTfidfProcessor']