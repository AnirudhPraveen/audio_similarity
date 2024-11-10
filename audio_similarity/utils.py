"""
Utility functions and classes for audio processing and data handling.

This module provides helper functions and classes used throughout the library
for common tasks such as audio validation, batch processing, and data
transformation.
"""

from pathlib import Path
from typing import List, Optional, Union, Iterator, Dict
from dataclasses import dataclass
import logging
import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)

@dataclass
class AudioBatch:
    """
    Container for batch processing of audio files.

    Parameters
    ----------
    waveforms : torch.Tensor
        Batch of audio waveforms
    sample_rates : List[int]
        Sample rates for each waveform
    file_paths : List[Path]
        Original file paths for each audio file

    Examples
    --------
    >>> batch = AudioBatch.from_files(['audio1.wav', 'audio2.wav'])
    >>> print(batch.waveforms.shape)
    torch.Size([2, 1, 16000])
    """
    
    waveforms: torch.Tensor
    sample_rates: List[int]
    file_paths: List[Path]

    @classmethod
    def from_files(
        cls,
        file_paths: List[Union[str, Path]],
        target_sample_rate: int = 16000,
        max_duration: Optional[float] = None
    ) -> 'AudioBatch':
        """
        Create a batch from a list of audio files.

        Parameters
        ----------
        file_paths : List[Union[str, Path]]
            List of paths to audio files
        target_sample_rate : int, optional
            Target sample rate for all audio, by default 16000
        max_duration : float, optional
            Maximum duration in seconds to load

        Returns
        -------
        AudioBatch
            Batch containing processed audio files

        Raises
        ------
        RuntimeError
            If any audio file fails to load
        """
        waveforms = []
        sample_rates = []
        valid_paths = []

        for path in file_paths:
            try:
                waveform, sample_rate = load_and_process_audio(
                    path,
                    target_sample_rate,
                    max_duration
                )
                waveforms.append(waveform)
                sample_rates.append(sample_rate)
                valid_paths.append(Path(path))
            except Exception as e:
                logger.warning(f"Failed to load {path}: {str(e)}")

        if not waveforms:
            raise RuntimeError("No valid audio files were loaded")

        # Pad to same length
        max_length = max(w.shape[-1] for w in waveforms)
        padded_waveforms = []
        for waveform in waveforms:
            padding = max_length - waveform.shape[-1]
            padded = torch.nn.functional.pad(waveform, (0, padding))
            padded_waveforms.append(padded)

        return cls(
            waveforms=torch.stack(padded_waveforms),
            sample_rates=sample_rates,
            file_paths=valid_paths
        )

def ensure_valid_audio(
    file_path: Union[str, Path],
    min_duration: float = 0.1,
    max_duration: Optional[float] = None
) -> bool:
    """
    Check if an audio file is valid and meets duration requirements.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to audio file
    min_duration : float, optional
        Minimum duration in seconds, by default 0.1
    max_duration : float, optional
        Maximum duration in seconds, by default None

    Returns
    -------
    bool
        True if audio file is valid

    Notes
    -----
    Checks for:
    - File existence
    - File can be opened
    - Duration requirements
    - Valid audio data
    """
    try:
        info = torchaudio.info(str(file_path))
        duration = info.num_frames / info.sample_rate

        if duration < min_duration:
            logger.warning(f"Audio file {file_path} is too short: {duration:.2f}s")
            return False

        if max_duration and duration > max_duration:
            logger.warning(f"Audio file {file_path} is too long: {duration:.2f}s")
            return False

        return True

    except Exception as e:
        logger.warning(f"Invalid audio file {file_path}: {str(e)}")
        return False

def load_and_process_audio(
    file_path: Union[str, Path],
    target_sample_rate: int = 16000,
    max_duration: Optional[float] = None
) -> tuple[torch.Tensor, int]:
    """
    Load and preprocess an audio file.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to audio file
    target_sample_rate : int, optional
        Target sample rate, by default 16000
    max_duration : float, optional
        Maximum duration in seconds to load

    Returns
    -------
    Tuple[torch.Tensor, int]
        Processed waveform and sample rate

    Raises
    ------
    ValueError
        If the audio file is invalid
    """
    if not ensure_valid_audio(file_path):
        raise ValueError(f"Invalid audio file: {file_path}")

    waveform, sample_rate = torchaudio.load(str(file_path))

    # Convert to mono if necessary
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if necessary
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
        waveform = resampler(waveform)
        sample_rate = target_sample_rate

    # Trim to max duration if specified
    if max_duration:
        max_length = int(max_duration * sample_rate)
        if waveform.shape[-1] > max_length:
            waveform = waveform[..., :max_length]

    return waveform, sample_rate

def get_audio_statistics(
    file_paths: List[Union[str, Path]]
) -> Dict[str, float]:
    """
    Calculate statistics for a collection of audio files.

    Parameters
    ----------
    file_paths : List[Union[str, Path]]
        List of paths to audio files

    Returns
    -------
    Dict[str, float]
        Dictionary containing statistics:
        - mean_duration
        - std_duration
        - total_duration
        - num_files
        - min_duration
        - max_duration

    Examples
    --------
    >>> stats = get_audio_statistics(['audio1.wav', 'audio2.wav'])
    >>> print(f"Mean duration: {stats['mean_duration']:.2f}s")
    """
    durations = []

    for path in file_paths:
        try:
            info = torchaudio.info(str(path))
            duration = info.num_frames / info.sample_rate
            durations.append(duration)
        except Exception as e:
            logger.warning(f"Failed to get info for {path}: {str(e)}")

    if not durations:
        return {
            "mean_duration": 0.0,
            "std_duration": 0.0,
            "total_duration": 0.0,
            "num_files": 0,
            "min_duration": 0.0,
            "max_duration": 0.0
        }

    durations = np.array(durations)
    return {
        "mean_duration": float(np.mean(durations)),
        "std_duration": float(np.std(durations)),
        "total_duration": float(np.sum(durations)),
        "num_files": len(durations),
        "min_duration": float(np.min(durations)),
        "max_duration": float(np.max(durations))
    }