"""
Visualization and benchmarking tools for audio similarity search.

This module provides tools for:
- Visualizing search results
- Benchmarking different FAISS indices
- Comparing index performance
- Generating performance reports
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Any
import time
import logging
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import faiss
from tqdm import tqdm

from .audio_processor import AudioProcessor, AudioTfidfProcessor
from .index_factory import IndexFactory, IndexType
from .utils import AudioBatch, ensure_valid_audio

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Container for index benchmark results."""
    index_type: str
    build_time: float
    query_time: float
    memory_usage: float
    recall: float
    k: int
    num_samples: int
    params: Dict[str, Any]

class AudioSimilaritySearch:
    """
    Main class for audio similarity search with visualization capabilities.

    Parameters
    ----------
    index_type : IndexType
        Type of FAISS index to use
    index_params : Dict[str, Any], optional
        Parameters for the chosen index type
    cache_dir : str, optional
        Directory for caching models and data
    device : str, optional
        Device to use for computation ('cuda' or 'cpu')
    """

    def __init__(
        self,
        index_type: IndexType = IndexType.FLAT,
        index_params: Optional[Dict[str, Any]] = None,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None
    ):
        # self.audio_processor = AudioProcessor(cache_dir=cache_dir, device=device)
        self.audio_processor = AudioProcessor(cache_dir=cache_dir, device=device)
        self.index_params = index_params or {}
        self.dimension = 768  # wav2vec2 embedding dimension
        
        # Initialize FAISS index
        self.index = IndexFactory.create_index(
            self.dimension,
            index_type,
            **self.index_params
        )
        
        self.benchmark_results = []
        self.file_paths = {}
        self.next_index = 0

    def add_audio(self, audio_path: Union[str, Path]) -> int:
        """
        Add a single audio file to the index.

        Parameters
        ----------
        audio_path : Union[str, Path]
            Path to audio file

        Returns
        -------
        int
            Index of the added audio file
        """
        if not ensure_valid_audio(audio_path):
            raise ValueError(f"Invalid audio file: {audio_path}")

        embedding = self.audio_processor.process_file(audio_path)
        self.index.add(embedding)
        
        current_index = self.next_index
        self.file_paths[current_index] = str(audio_path)
        self.next_index += 1
        
        return current_index

    def add_batch(self, audio_paths: List[Union[str, Path]]) -> List[int]:
        """
        Add multiple audio files to the index.
        Parameters
        ----------
        audio_paths : List[Union[str, Path]]
            List of paths to audio files
        Returns
        -------
        List[int]
            Indices of added files
        """
        print(f"Starting add_batch with {len(audio_paths)} files")  
        indices = []
        
        print("Creating AudioBatch...")  
        batch = AudioBatch.from_files(audio_paths)
        
        embeddings = []
        valid_paths = []
        for i, (waveform, path) in enumerate(zip(batch.waveforms, batch.file_paths)):
            print(f"Processing file {i+1}/{len(batch.waveforms)}: {path}")  
            print(f"Waveform shape: {waveform.shape}")  
            try:
                embedding = self.audio_processor.get_embedding(waveform)
                print(f"Generated embedding shape: {embedding.shape}")  
                embeddings.append(embedding)
                valid_paths.append(path)
            except Exception as e:
                print(f"Full error details: {str(e)}")  
                logger.warning(f"Failed to process {path}: {e}")
        
        print(f"Successfully processed {len(embeddings)} files")  
        
        if embeddings:
            print("Stacking embeddings...") 
            embeddings = np.vstack(embeddings)
            print(f"Adding to index, embeddings shape: {embeddings.shape}") 
            self.index.add(embeddings)
            
            for path in valid_paths:
                indices.append(self.next_index)
                self.file_paths[self.next_index] = str(path)
                self.next_index += 1
                print(f"Added file {path} with index {self.next_index-1}") 
        
        print(f"Returning {len(indices)} indices") 
        return indices 

    def search(
        self,
        query_path: Union[str, Path],
        k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Search for similar audio files.

        Parameters
        ----------
        query_path : Union[str, Path]
            Path to query audio file
        k : int, optional
            Number of results to return

        Returns
        -------
        List[Tuple[str, float]]
            List of (file_path, distance) tuples
        """
        query_embedding = self.audio_processor.process_file(query_path)
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx in self.file_paths:
                results.append((self.file_paths[idx], float(dist)))
        
        return results

    # def visualize_search_results(
    #     self,
    #     query_path: str,
    #     results: List[Tuple[str, float]],
    #     save_path: Optional[str] = None,
    #     show: bool = True
    # ) -> Optional[plt.Figure]:
    #     """
    #     Visualize search results with distance comparison.

    #     Parameters
    #     ----------
    #     query_path : str
    #         Path to query audio file
    #     results : List[Tuple[str, float]]
    #         List of (file_path, distance) tuples
    #     save_path : str, optional
    #         Path to save the visualization
    #     show : bool, optional
    #         Whether to display the plot
    #     """
    #     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
    #     # Plot query waveform
    #     query_waveform, _ = self.audio_processor.load_audio(query_path)
    #     ax1.plot(query_waveform[0].numpy())
    #     ax1.set_title('Query Audio Waveform')
    #     ax1.set_xlabel('Sample')
    #     ax1.set_ylabel('Amplitude')
        
    #     # Plot distances
    #     distances = [dist for _, dist in results]
    #     labels = [Path(path).name for path, _ in results]
        
    #     sns.barplot(x=labels, y=distances, ax=ax2)
    #     ax2.set_title('Distance to Query')
    #     ax2.set_xlabel('Similar Audio Files')
    #     ax2.set_ylabel('Distance')
    #     ax2.tick_params(axis='x', rotation=45)
        
    #     plt.tight_layout()
        
    #     if save_path:
    #         plt.savefig(save_path, bbox_inches='tight')
        
    #     if show:
    #         plt.show()
    #         return None
    #     return fig

    def visualize_search_results(
        self,
        query_path: str,
        results: List[Tuple[str, float]],
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Optional[plt.Figure]:
        """
        Visualize search results with distance comparison.
        Parameters
        ----------
        query_path : str
            Path to query audio file
        results : List[Tuple[str, float]]
            List of (file_path, distance) tuples
        save_path : str, optional
            Path to save the visualization
        show : bool, optional
            Whether to display the plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot query waveform
        query_waveform, sr = self.audio_processor.load_audio(query_path)  # Now correctly getting sr
        waveform = query_waveform[0].numpy()
        duration = len(waveform) / sr  # Calculate duration in seconds
        time_axis = np.linspace(0, duration, len(waveform))  # Create time axis
        
        ax1.plot(time_axis, waveform)
        ax1.set_title('Query Audio Waveform')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Amplitude')
        
        # Add grid for better readability
        ax1.grid(True, alpha=0.3)
        
        # Format x-axis ticks to show reasonable time intervals
        ax1.xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        
        # Plot distances
        distances = [dist for _, dist in results]
        labels = [Path(path).name for path, _ in results]
        
        # Create barplot
        bars = sns.barplot(x=labels, y=distances, ax=ax2)
        
        # Add value annotations on top of each bar
        for i, bar in enumerate(bars.patches):
            bars.text(
                bar.get_x() + bar.get_width()/2.,  # x position
                bar.get_height(),                  # y position
                f'{distances[i]:.4f}',            # text (distance value with 4 decimal places)
                ha='center',                      # horizontal alignment
                va='bottom'                       # vertical alignment
            )
        
        # Improve y-axis visibility
        y_min = min(distances)
        y_max = max(distances)
        y_padding = (y_max - y_min) * 0.1  # Add 10% padding
        
        ax2.set_ylim(y_min - y_padding, y_max + y_padding)
        
        # Format y-axis to show more decimal places
        ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
        
        ax2.set_title('Distance to Query')
        ax2.set_xlabel('Similar Audio Files')
        ax2.set_ylabel('Distance')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        if show:
            plt.show()
            return None
        return fig

    def save(self, directory: Union[str, Path]):
        """
        Save the index and file mappings.

        Parameters
        ----------
        directory : Union[str, Path]
            Directory to save the index and mappings
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(directory / "index.faiss"))
        
        # Save file mappings
        mappings_path = directory / "mappings.json"
        with open(mappings_path, 'w') as f:
            pd.Series(self.file_paths).to_json(f)

    def load(self, directory: Union[str, Path]):
        """
        Load the index and file mappings.

        Parameters
        ----------
        directory : Union[str, Path]
            Directory containing the saved index and mappings
        """
        directory = Path(directory)
        
        # Load FAISS index
        self.index = faiss.read_index(str(directory / "index.faiss"))
        
        # Load file mappings
        mappings_path = directory / "mappings.json"
        with open(mappings_path, 'r') as f:
            mappings = pd.read_json(f, typ='series')
            self.file_paths = {int(k): str(v) for k, v in mappings.items()}
            self.next_index = max(self.file_paths.keys()) + 1 if self.file_paths else 0

    def benchmark(
        self,
        compare_with: List[Dict[str, Any]],
        num_samples: int = 10000,
        num_queries: int = 100,
        k: int = 5,
        embeddings: Optional[np.ndarray] = None
    ) -> List[BenchmarkResult]:
        """
        Benchmark different index configurations.

        Parameters
        ----------
        compare_with : List[Dict[str, Any]]
            List of dicts with 'type' (IndexType) and 'params' (dict)
        num_samples : int, optional
            Number of samples for benchmarking
        num_queries : int, optional
            Number of query samples
        k : int, optional
            Number of nearest neighbors
        embeddings : np.ndarray, optional
            Pre-computed embeddings to use

        Returns
        -------
        List[BenchmarkResult]
            Benchmark results for each index configuration
        """
        if embeddings is None:
            embeddings = np.random.random(
                (num_samples, self.dimension)
            ).astype('float32')
            
        query_data = np.random.random(
            (num_queries, self.dimension)
        ).astype('float32')
        
        # Compute ground truth
        flat_index = faiss.IndexFlatL2(self.dimension)
        flat_index.add(embeddings)
        _, ground_truth = flat_index.search(query_data, k)
        
        results = []
        for config in tqdm(compare_with, desc="Benchmarking indices"):
            index_type = config['type']
            params = config.get('params', {})
            
            # Create and train index
            start_time = time.time()
            index = IndexFactory.create_index(self.dimension, index_type, **params)
            
            if getattr(index, 'train_required', False):
                index.train(embeddings)
            index.add(embeddings)
            build_time = time.time() - start_time
            
            # Query
            start_time = time.time()
            _, result_ids = index.search(query_data, k)
            query_time = (time.time() - start_time) / num_queries
            
            # Compute metrics
            recall = self._compute_recall(ground_truth, result_ids, k)
            memory_usage = index.ntotal * self.dimension * 4 / (1024 * 1024)  # MB
            
            result = BenchmarkResult(
                index_type=index_type.value,
                build_time=build_time,
                query_time=query_time,
                memory_usage=memory_usage,
                recall=recall,
                k=k,
                num_samples=num_samples,
                params=params
            )
            results.append(result)
            self.benchmark_results.append(result)
        
        return results

    # @staticmethod
    # def _compute_recall(
    #     ground_truth: np.ndarray,
    #     results: np.ndarray,
    #     k: int
    # ) -> float:
    #     """Compute recall@k between ground truth and results."""
    #     n = ground_truth.shape[0]
    #     recall = 0
    #     for i in range(n):
    #         gt = set(ground_truth[i, :k])
    #         res = set(results[i, :k])
    #         recall += len(gt.intersection(res)) / k
    #     return recall / n