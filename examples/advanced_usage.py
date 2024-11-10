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
    parser.add_argument(
        "--query",
        type=str,
        help="Path to the query audio file"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of similar results to return"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark comparisons of different index types"
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

def verify_query_file(query_path: str):
    """Verify that the query file exists and is a supported audio format."""
    supported_formats = {'.wav', '.mp3', '.flac', '.ogg'}
    query_path = Path(query_path)
    
    if not query_path.exists():
        raise FileNotFoundError(f"Query file {query_path} does not exist")
    
    if query_path.suffix.lower() not in supported_formats:
        raise ValueError(f"Query file format {query_path.suffix} not supported. Supported formats: {supported_formats}")
    
    return query_path

def compare_indices(audio_files, output_dir: Path):
    """Compare different FAISS index types."""
    configs = [
        {'type': IndexType.FLAT},
        {'type': IndexType.IVF, 'params': {'nlist': 10}},
        {'type': IndexType.HNSW, 'params': {'M': 16}},
        {'type': IndexType.PQ, 'params': {'M': 8, 'nbits': 8}},
    ]
    
    searcher = AudioSimilaritySearch(index_type=IndexType.FLAT)
    
    print("Indexing files for benchmark...")
    searcher.add_batch(audio_files)
    
    print("\nBenchmarking different index types...")
    results = searcher.benchmark(
        compare_with=configs,
        num_samples=1000,
        num_queries=10,
        k=5
    )
    
    output_dir.mkdir(exist_ok=True, parents=True)
    searcher.visualize_benchmarks(
        metric="all",
        save_path=output_dir / "benchmark_results.png"
    )
    
    return results

def search_similar_audio(audio_files, query_file: Path, top_k: int = 5):
    """Search for similar audio files to the query."""
    searcher = AudioSimilaritySearch(
        index_type=IndexType.IVF,
        index_params={'nlist': 10}
    )
    
    # Index all files
    print("Indexing audio files...")
    searcher.add_batch(audio_files)
    
    # Perform search
    print(f"\nSearching for top {top_k} matches to {query_file.name}...")
    results = searcher.search(query_file, k=top_k)
    
    return results

def main():
    """Run advanced example with custom dataset location and query."""
    # Parse command line arguments
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Convert string paths to Path objects
    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    
    print(f"Using dataset from: {dataset_path}")
    print(f"Outputs will be saved to: {output_dir}")
    
    # Load dataset
    print("\n1. Loading audio files...")
    try:
        audio_files = load_audio_files(dataset_path)
        print(f"Found {len(audio_files)} audio files")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return
    
    # Run benchmark if requested
    if args.benchmark:
        print("\n2. Running index type comparison benchmark...")
        benchmark_results = compare_indices(audio_files, output_dir)
        
        print("\nBenchmark Results:")
        print("-" * 50)
        for result in benchmark_results:
            print(f"\nIndex Type: {result.index_type}")
            print(f"Build Time: {result.build_time:.4f}s")
            print(f"Query Time: {result.query_time:.6f}s")
            print(f"Memory Usage: {result.memory_usage:.2f}MB")
            print(f"Recall@{result.k}: {result.recall:.4f}")
    
    # Process query if provided
    if args.query:
        try:
            query_file = verify_query_file(args.query)
            results = search_similar_audio(audio_files, query_file, args.top_k)
            
            print("\nSearch Results:")
            print("-" * 50)
            for i, (file_path, distance) in enumerate(results, 1):
                print(f"{i}. File: {Path(file_path).name}")
                print(f"   Distance: {distance:.4f}")
                
        except (FileNotFoundError, ValueError) as e:
            print(f"Error with query file: {e}")
            return

if __name__ == "__main__":
    main()