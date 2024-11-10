import numpy as np
import torch
import torchaudio
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from audio_similarity import AudioSimilaritySearch, IndexType

def setup_argparse():
    """Set up command line argument parsing."""
    parser = argparse.ArgumentParser(description="Audio similarity search with custom dataset location")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the directory containing audio files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory for storing output files and visualizations"
    )
    return parser

def load_audio_files(dataset_path: Path):
    """Load audio files from the specified directory."""
    supported_formats = {'.wav', '.mp3', '.flac', '.ogg'}
    audio_files = []
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")
    
    for file_path in dataset_path.rglob('*'):
        if file_path.suffix.lower() in supported_formats:
            audio_files.append(file_path)
    
    if not audio_files:
        raise ValueError(f"No audio files found in {dataset_path}. Please provide a directory containing audio files.")
    
    return audio_files

def compare_indices(audio_files, output_dir: Path):
    """Compare different FAISS index types."""
    # Define index configurations to compare
    configs = [
        {'type': IndexType.FLAT},
        {'type': IndexType.IVF, 'params': {'nlist': 10}},
        {'type': IndexType.HNSW, 'params': {'M': 16}},
        {'type': IndexType.PQ, 'params': {'M': 8, 'nbits': 8}},
    ]
    
    # Initialize searcher with flat index for ground truth
    searcher = AudioSimilaritySearch(index_type=IndexType.FLAT)
    
    # Index files
    print("Indexing files for benchmark...")
    searcher.add_batch(audio_files)
    
    # Run benchmark
    print("\nBenchmarking different index types...")
    results = searcher.benchmark(
        compare_with=configs,
        num_samples=1000,
        num_queries=10,
        k=5
    )
    
    # Visualize benchmark results
    output_dir.mkdir(exist_ok=True, parents=True)
    searcher.visualize_benchmarks(
        metric="all",
        save_path=output_dir / "benchmark_results.png"
    )
    
    return results

def batch_processing_example(audio_files):
    """Demonstrate batch processing capabilities."""
    searcher = AudioSimilaritySearch(
        index_type=IndexType.IVF,
        index_params={'nlist': 10}
    )
    
    # Process in batches
    batch_size = 3
    for i in range(0, len(audio_files), batch_size):
        batch = audio_files[i:i + batch_size]
        searcher.add_batch(batch)
    
    # Perform multiple searches
    all_results = []
    for query_file in tqdm(audio_files[:3], desc="Searching"):
        results = searcher.search(query_file, k=3)
        all_results.append(results)
    
    return all_results

def main():
    """Run advanced example with custom dataset location."""
    # Parse command line arguments
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Convert string paths to Path objects
    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    
    print(f"Using dataset from: {dataset_path}")
    print(f"Outputs will be saved to: {output_dir}")
    
    print("\n1. Loading audio files...")
    try:
        audio_files = load_audio_files(dataset_path)
        print(f"Found {len(audio_files)} audio files")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return
    
    print("\n2. Comparing different index types...")
    benchmark_results = compare_indices(audio_files, output_dir)
    
    print("\nBenchmark Results:")
    print("-" * 50)
    for result in benchmark_results:
        print(f"\nIndex Type: {result.index_type}")
        print(f"Build Time: {result.build_time:.4f}s")
        print(f"Query Time: {result.query_time:.6f}s")
        print(f"Memory Usage: {result.memory_usage:.2f}MB")
        print(f"Recall@{result.k}: {result.recall:.4f}")
    
    print("\n3. Demonstrating batch processing...")
    search_results = batch_processing_example(audio_files)
    
    print("\nBatch Processing Results:")
    print("-" * 50)
    for i, results in enumerate(search_results):
        print(f"\nQuery {i+1} results:")
        for j, (file_path, distance) in enumerate(results[:3], 1):
            print(f"{j}. File: {Path(file_path).name}")
            print(f"   Distance: {distance:.4f}")
    
    print(f"\nDone! Check the benchmark visualization in {output_dir}")

if __name__ == "__main__":
    main()